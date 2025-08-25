import os

import torch
from torch.utils.cpp_extension import load

# 编译并加载 CUDA 扩展
sign_xor = load(
    name="sign_xor",
    sources=["sign_xor_kernel.cu"],
    extra_include_paths=[os.environ["CONDA_PREFIX"]],
    verbose=True
)

# 造测试数据
coord1 = torch.randint(-10, 10, (8,1,256,1,3), dtype=torch.int64, device="cuda")
coord2 = torch.randint(-10, 10, (8,256,1,325,3), dtype=torch.int64, device="cuda")

# 调用自定义 CUDA kernel
out = sign_xor.sign_xor_cuda(coord1, coord2)
print(out.shape, out.dtype)
