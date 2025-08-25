#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sign_xor_kernel(
    const long* __restrict__ coord1,
    const long* __restrict__ coord2,
    short* __restrict__ output,
    int64_t N // 总元素数
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        long a = coord1[idx];
        long b = coord2[idx];
        // 取符号 bit (>=0 → 0, <0 → 1)
        int sa = (a >= 0) ? 0 : 1;
        int sb = (b >= 0) ? 0 : 1;
        int x = sa ^ sb; // 异或
        output[idx] = (short)(1 - (x << 1)); // {+1,-1}
    }
}

torch::Tensor sign_xor_cuda(torch::Tensor coord1, torch::Tensor coord2) {
    auto out = torch::empty_like(coord1.expand_as(coord2), torch::dtype(torch::kInt16));
    int64_t N = out.numel();

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    sign_xor_kernel<<<blocks, threads>>>(
        coord1.data_ptr<long>(),
        coord2.data_ptr<long>(),
        out.data_ptr<short>(),
        N
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sign_xor_cuda", &sign_xor_cuda, "Sign-Xor kernel (CUDA)");
}

