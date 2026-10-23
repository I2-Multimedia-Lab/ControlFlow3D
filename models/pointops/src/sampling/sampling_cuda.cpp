#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include "sampling_cuda_kernel.h"


void furthestsampling_cuda(int b, int n, int m, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor)
{
    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    furthestsampling_cuda_launcher(b, n, m, points, temp, idx);
}
