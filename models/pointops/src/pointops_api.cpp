#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ballquery/ballquery_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"
#include "knnquery_heap/knnquery_heap_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ballquery_cuda", &ballquery_cuda_fast, "ballquery_cuda_fast");   // Bind the Python name, C++ function and description.

    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");

    m.def("knnquery_heap_cuda", &knnquery_heap_cuda, "knnquery_heap_cuda");
}
