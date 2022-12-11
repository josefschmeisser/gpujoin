

cmake invocation:

cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_C_COMPILER="/usr/bin/gcc" -D CMAKE_CXX_COMPILER="/usr/bin/g++" -D CMAKE_CUDA_COMPILER=/usr/bin/nvcc -D CMAKE_CUDA_FLAGS=" -gencode=arch=compute_70,code=sm_70" ..
