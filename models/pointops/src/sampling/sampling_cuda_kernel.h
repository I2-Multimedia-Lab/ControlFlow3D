#ifndef _SAMPLING_CUDA_KERNEL
#define _SAMPLING_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void furthestsampling_cuda(int b, int n, int m, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void furthestsampling_cuda_launcher(int b, int n, int m, const float *dataset, float *temp, int *idxs);

#ifdef __cplusplus
}
#endif
#endif
