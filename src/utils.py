import time
import inspect
import functools
from typing import Callable, List, Tuple, Optional, Literal, Dict, Sequence, Any
from contextlib import ContextDecorator
import torch

Mark = Callable[[str], None]

def timeline(
    name: Optional[str] = None,
    *,
    unit: Literal["s", "ms", "us"] = "ms",
    clock: Literal["perf", "process"] = "perf",
    printer: Callable[[str], None] = print,
    show_deltas: bool = True,    # 显示相邻标记的间隔
    show_total: bool = True,     # 显示总耗时
):
    """
    装饰器：被装饰函数内部可调用 mark(label) 记录时间点，返回时统一输出时间线。
    使用方法：
        @timeline("fit-epoch")
        def train(..., mark=None):
            ...; mark("load"); ...; mark("forward"); ...; mark("backward")
    """
    get_time = time.perf_counter if clock == "perf" else time.process_time

    def _fmt(sec: float) -> str:
        if unit == "s":
            return f"{sec:.6f}s"
        if unit == "ms":
            return f"{sec*1e3:.3f}ms"
        if unit == "us":
            return f"{sec*1e6:.0f}µs"
        return f"{sec:.6f}s"

    def decorator(func):
        disp = name or f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                marks: List[Tuple[str, float]] = []
                t0 = get_time()

                def mark(label: str):
                    marks.append((label, get_time()))

                # 将 mark 注入到函数参数里（若用户没显式传入）
                if "mark" not in kwargs:
                    kwargs["mark"] = mark

                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    t_end = get_time()
                    # 组装输出
                    lines = [f"[timeline] {disp}"]
                    prev = t0
                    for lbl, tt in marks:
                        delta = tt - prev
                        since0 = tt - t0
                        if show_deltas:
                            lines.append(f"  - {lbl}: +{_fmt(delta)} (t={_fmt(since0)})")
                        else:
                            lines.append(f"  - {lbl}: t={_fmt(since0)}")
                        prev = tt
                    if show_total:
                        lines.append(f"  = total: {_fmt(t_end - t0)}")
                    printer("\n".join(lines))
            return awrapper
        else:
            @functools.wraps(func)
            def swrapper(*args, **kwargs):
                marks: List[Tuple[str, float]] = []
                t0 = get_time()

                def mark(label: str):
                    marks.append((label, get_time()))

                if "mark" not in kwargs:
                    kwargs["mark"] = mark

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    t_end = get_time()
                    lines = [f"[timeline] {disp}"]
                    prev = t0
                    for lbl, tt in marks:
                        delta = tt - prev
                        since0 = tt - t0
                        if show_deltas:
                            lines.append(f"  - {lbl}: +{_fmt(delta)} (t={_fmt(since0)})")
                        else:
                            lines.append(f"  - {lbl}: t={_fmt(since0)}")
                        prev = tt
                    if show_total:
                        lines.append(f"  = total: {_fmt(t_end - t0)}")
                    printer("\n".join(lines))
            return swrapper

    return decorator


def named(**tensors: torch.Tensor) -> Dict[str, torch.Tensor]:
    return tensors

def pinned_copy_by_name(named_tensors: dict[str, torch.Tensor], _cache: dict[str, torch.Tensor] = {}):
    out: dict[str, torch.Tensor] = {}
    for name, t in named_tensors.items():
        # 若规格改变则重建
        if (name not in _cache or
            _cache[name].shape != t.shape or
            _cache[name].dtype != t.dtype):
            _cache[name] = torch.empty_like(t, device="cpu", pin_memory=True)
        dst = _cache[name]
        dst.copy_(t, non_blocking=True)
        out[name] = dst
    return out



def batch_concat(
    base: torch.Tensor,          # (bs1, A, B, ...)
    small: torch.Tensor,         # (bs2, a, b, ...)
    pad_value: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 base 与 small 合并为 (bs1+bs2, A, B, ...)，其中 small 的每个样本
    左对齐放入 (A,B,...) 的画布，右/下/后侧用 pad_value 填充。
    返回 (y, mask)，mask 为 bool，True 表示有效原数据位置（方案 B）。

    约束：
      - base.ndim == small.ndim
      - base.shape[1:] = (A,B,...)，small.shape[1:] = (a,b,...)
      - 且每维 A>=a, B>=b, ...
      - dtype 与 device 以 base 为准（small 会被拷贝到相同 device/dtype）
    """
    if base.ndim != small.ndim:
        raise ValueError(f"rank mismatch: base.ndim={base.ndim}, small.ndim={small.ndim}")
    if base.ndim < 2:
        raise ValueError("expect at least 2 dims: (batch, ...)")
    bs1, *big_shape = base.shape
    bs2, *small_shape = small.shape

    if any(B < S for B, S in zip(big_shape, small_shape)):
        raise ValueError(f"target dims must be >= small dims, got {big_shape} vs {small_shape}")

    # 统一 dtype/device 到 base
    if small.dtype != base.dtype or small.device != base.device:
        small = small.to(dtype=base.dtype, device=base.device)

    # 分配输出与 mask（方案B：有效为 True）
    out_shape = (bs1 + bs2, *big_shape)
    y = torch.full(out_shape, pad_value, dtype=base.dtype, device=base.device)
    mask = torch.zeros(out_shape, dtype=torch.bool, device=base.device)

    # 1) 复制 base 批
    y[:bs1] = base
    mask[:bs1] = True  # base 全部为有效

    # 2) 将 small 批逐样本放入左对齐画布
    # 写入区域：[:bs2, :a, :b, ...]
    write_slices = [slice(bs1, bs1 + bs2)]
    valid_slices = [slice(0, bs2)]
    for S in small_shape:
        write_slices.append(slice(0, S))
        valid_slices.append(slice(0, S))
    write_slices = tuple(write_slices)
    valid_slices = tuple(valid_slices)

    # 放置数据
    y[write_slices] = small
    mask[write_slices] = True  # 仅 small 的有效子区为 True；填充区域保持 False

    return y, mask

def to_dev(
    *tensors: torch.Tensor,
    device: torch.device | str = 'cpu',
    s: Optional[int] = None,
    e: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    将若干张量的 dim=0 切片 [s:e] 非阻塞传输到指定 device，并按顺序返回。
    - 若 s、e 为 None，则等价于整段 [:]
    - 要求所有输入是 torch.Tensor
    """
    dev = torch.device(device)
    sliced = tuple(t.__getitem__(slice(s, e)) for t in tensors)  # 避免触发高级索引
    return tuple(t.to(dev, non_blocking=True) for t in sliced)

def get_strategy(loss_type: dict, epoch: int) -> dict:
    # 找到所有小于等于 epoch 的阈值
    valid_keys = [k for k in loss_type.keys() if epoch <= k]
    if not valid_keys:
        # 若没有上界覆盖，则可选择返回最大键对应配置或抛错
        raise ValueError(f"No strategy configured for epoch={epoch}")
    # 取最小的上界（即最早满足条件的区间）
    key = min(valid_keys)
    return loss_type[key]