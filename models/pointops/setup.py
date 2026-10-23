from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointops',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',

            'src/ballquery/ballquery_cuda.cpp',
            'src/ballquery/ballquery_cuda_kernel.cu',

            'src/knnquery_heap/knnquery_heap_cuda.cpp',
            'src/knnquery_heap/knnquery_heap_cuda_kernel.cu',

            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',
        ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
