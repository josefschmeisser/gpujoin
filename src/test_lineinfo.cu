#include <cstdio>

#include "tpch/common.hpp"

template<class T>
__global__ void cuda_hello(T var){
    printf("Hello World from GPU!\n");
}

int main() {
    Database db;
    sort_relation(db.part);
    cuda_hello<<<1,1>>>(1);
    cudaDeviceSynchronize();
    return 0;
}
