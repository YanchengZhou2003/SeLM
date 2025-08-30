import functools
import inspect
import math
import time
from contextlib import ContextDecorator
from typing import (Any, Callable, Dict, Iterable, List, Literal, Optional,
                    Sequence, Tuple)

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


def fmt6(x):
    sgn = 1 if x < 0 else 0
    absx = abs(x)
    int_len = len(str(int(absx)))
    # 留出小数点和可能的负号
    decimals = 6 - int_len - 1 - sgn
    if decimals < 0:
        # 放不下，返回溢出标记或科学计数法（任选其一）
        # return "######"  # 固定 6 个井号
        # 或者用科学计数法，尽量贴近 6 宽：
        return f"{x:.1e}"[:6]  # 简单截断
    return f"{x:.{decimals}f}"

def fmt6w(x):
    s = fmt6(x)
    # 若 s 长度不足 6，用空格左填充至 6
    return f"{s:>6}"[:6]


def js_div(
    input,                 # P：log-prob 或 prob
    target,                # Q：log-prob 或 prob
    log_input: bool = True,
    log_target: bool = False,
    reduction: str = 'none',  # 与 F.kl_div 一致的默认
    eps: float = 1e-12
):
    """
    返回与 input/target 同形的逐元素 JS 的“密度项”，
    需要调用方沿分布维 sum（与 F.kl_div 的使用一致）。
    说明：这里的“逐元素”指把 KL 的 integrand 展开到分布维上。
    """
    # 统一为 log 概率
    if log_input:
        logP = input
    else:
        P = input.clamp_min(eps)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
        logP = P.log()

    if log_target:
        logQ = target
    else:
        Q = target.clamp_min(eps)
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(eps)
        logQ = Q.log()

    # logM = log(0.5*P + 0.5*Q)
    logM = torch.logaddexp(logP, logQ) - math.log(2.0)

    # 逐元素 KL integrand：P*(logP-logM) 与 Q*(logQ-logM)
    P = torch.exp(logP)
    Q = torch.exp(logQ)
    elem_kl_p_m = P * (logP - logM)
    elem_kl_q_m = Q * (logQ - logM)

    # 逐元素 JS integrand 的和（还没沿分布维 sum）
    elem_js = 0.5 * (elem_kl_p_m + elem_kl_q_m)  # 形状与 input 相同

    if reduction == 'none':
        return elem_js
    elif reduction == 'sum':
        return elem_js.sum()
    elif reduction == 'mean':
        return elem_js.mean()
    elif reduction == 'batchmean':
        # 约定 batch 在 0 维；与 F.kl_div 相同语义：先总和，再除以 batch 大小
        return elem_js.sum() / max(elem_js.size(0), 1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    
def is_all_ones(
    x: torch.Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 0.0,
    empty_is_one: bool = False
) -> Tuple[bool, torch.dtype]:
    """
    检查张量是否全为 1（根据其 dtype 语义），并返回 (是否全为1, dtype)。

    规则：
      - bool: 全为 True
      - 整数: 全等于 1
      - 浮点: 全接近 1（使用 rtol/atol）
      - 复数: 实部接近 1 且虚部接近 0（使用 rtol/atol）
      - 空张量: 返回 empty_is_one（默认 False）

    参数:
      x: 待检查的张量
      atol: 绝对误差容差（针对浮点/复数）
      rtol: 相对误差容差（针对浮点/复数）
      empty_is_one: 空张量是否视为全为 1

    返回:
      (is_all_ones: bool, dtype: torch.dtype)
    """
    dtype = x.dtype

    # 空张量处理
    if x.numel() == 0:
        return (bool(empty_is_one), dtype)

    if dtype == torch.bool:
        # 全为 True
        return (bool(torch.all(x)), dtype)

    if dtype.is_floating_point:
        # 使用 torch.isclose 与 1.0 比较
        one = torch.ones((), dtype=dtype, device=x.device)
        is_one = torch.isclose(x, one, rtol=rtol, atol=atol)
        return (bool(torch.all(is_one)), dtype)

    if dtype in (torch.complex64, torch.complex128, torch.complex32) if hasattr(torch, "complex32") else \
       (dtype == torch.complex64 or dtype == torch.complex128):
        # 复数：实部 ~ 1，虚部 ~ 0
        real_close = torch.isclose(x.real, torch.ones((), dtype=x.real.dtype, device=x.device), rtol=rtol, atol=atol)
        imag_close = torch.isclose(x.imag, torch.zeros((), dtype=x.imag.dtype, device=x.device), rtol=rtol, atol=atol)
        return (bool(torch.all(real_close & imag_close)), dtype)

    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        # 整数精确比较
        one = torch.ones((), dtype=dtype, device=x.device)
        return (bool(torch.all(x == one)), dtype)

    # 其他非常见 dtype（如 bfloat16、float16 等）会被 is_floating_point 捕获；
    # 如果出现未覆盖 dtype，这里保守返回 False。
    return (False, dtype)


from typing import Dict, Iterable, Tuple

import torch


def load_state_dict_skip_prefixes(
    model: torch.nn.Module,
    state_like: Dict[str, torch.Tensor],
    prefixes_to_skip: Iterable[str] = ("cte",),
    strict: bool = False,
    verbose: bool = True
) -> Tuple[Iterable[str], Iterable[str]]:
    """
    将 state_dict 加载到 model，但跳过指定前缀的所有键（如 'cte', 'cte.*'）。
    
    参数:
      model: 目标模型
      state_like: 可能是 state_dict 或 { 'state_dict': state_dict } 的对象
      prefixes_to_skip: 需要跳过的前缀集合；会同时匹配精确键与其子键（k == p 或 k.startswith(p + '.'))
      strict: 传给 model.load_state_dict 的 strict；通常建议 False
      verbose: 打印被跳过的键与 load_state 结果

    返回:
      (missing_keys, unexpected_keys)：与 nn.Module.load_state_dict 一致
    """
    # 1) 取出真正的 state_dict
    if "state_dict" in state_like and isinstance(state_like["state_dict"], dict):
        state_dict = state_like["state_dict"]
    else:
        state_dict = state_like

    # 2) 同时考虑 DDP 的 'module.' 前缀
    #    我们扩展跳过列表，加入 'module.<prefix>'
    expanded_prefixes = set(prefixes_to_skip)
    expanded_prefixes.update([f"module.{p}" for p in prefixes_to_skip])

    def should_skip(k: str) -> bool:
        for p in expanded_prefixes:
            if k == p or k.startswith(p + "."):
                return True
        return False

    # 3) 过滤 state_dict
    filtered = {k: v for k, v in state_dict.items() if not should_skip(k)}

    if verbose:
        skipped = [k for k in state_dict.keys() if should_skip(k)]
        if skipped:
            print(f"[load_state_dict_skip_prefixes] Skipped {len(skipped)} keys:")
            for k in skipped:
                print("  -", k)

    # 4) 加载
    load_result = model.load_state_dict(filtered, strict=strict)

    if verbose:
        missing, unexpected = load_result
        print("[load_state_dict_skip_prefixes] load_state result:")
        print("  missing_keys  :", missing)
        print("  unexpected_keys:", unexpected)

    return load_result

def fetch_locals(*names: str) -> Dict[str, Any]:
    """
    从调用者的局部作用域（local namespace）抓取指定变量名，并以字典返回。
    若变量不存在，将不会出现在结果中。
    """
    frame = inspect.currentframe()
    try:
        caller = frame.f_back  # 上一层调用者
        locs = caller.f_locals if caller is not None else {}
        return {name: locs[name] for name in names if name in locs}
    finally:
        # 避免循环引用导致的内存泄漏
        del frame
        if 'caller' in locals():
            del caller


def to_one_hot_logits(
    targets: torch.Tensor,  # (B, T) 的类别索引
    vocab_size: int,        # V
    *,
    pos_val: float = 1.0,   # 目标位置的值
    neg_val: float = -1e9,   # 非目标位置的值
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    将 (B, T) 的类别索引转换为 (B, T, V) 的 one-hot 风格“logits”。

    约定：
      - targets 中的取值应为 [0, V-1] 的整数（可为任意整型/长整型张量）。
      - 返回张量在 (b, t, targets[b,t]) 位置为 pos_val，其余为 neg_val。
      - 如果需要真正的 logits（例如对交叉熵无穷大间隔），可以设 pos_val=0, neg_val=-inf，
        然后在 softmax 前加上这些分数。

    参数:
      targets: (B, T) 整数索引
      vocab_size: 词表大小 V
      pos_val: 目标位置的值（默认 1.0）
      neg_val: 非目标位置的值（默认 0.0）
      dtype: 输出 dtype（默认跟随 pos_val/neg_val 推断或使用浮点）

    返回:
      (B, T, V) 的张量
    """
    if targets.dim() != 2:
        raise ValueError(f"targets must be 2D (B, T), got shape {tuple(targets.shape)}")

    B, T = targets.shape
    device = targets.device

    # 选择 dtype：优先使用用户指定；否则用浮点（与 pos/neg 值兼容）
    if dtype is None:
        # 若 pos/neg 是浮点，则用 float32；否则默认 float32
        dtype = torch.float32

    # 初始化为 neg_val
    out = torch.full((B, T, vocab_size), fill_value=neg_val, dtype=dtype, device=device)

    # 安全性检查：索引范围
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long):
        raise TypeError(f"targets must be integer dtype, got {targets.dtype}")

    if torch.any(targets < 0) or torch.any(targets >= vocab_size):
        bad_idx = torch.stack([(targets < 0), (targets >= vocab_size)]).any(0)
        # 定位第一个错误位置，给出更明确的报错
        first = (bad_idx.view(-1).nonzero(as_tuple=False)[0].item()
                 if bad_idx.any() else None)
        raise IndexError(f"targets contain out-of-range indices for V={vocab_size}. Example bad index at flat pos {first}")

    # 用高级索引写入 pos_val
    # 展平成 (B*T,) 方便构造坐标
    flat_targets = targets.reshape(-1)
    bt = torch.arange(B * T, device=device)
    b = bt // T
    t = bt % T
    out[b, t, flat_targets] = pos_val

    return out