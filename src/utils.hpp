#pragma once

#include <vector>

#include <cuda_runtime_api.h>
#include <numa.h>

struct vector_to_device_array {
    template<class T>
    T* operator() (const std::vector<T>& vec) {
        T* dst;
        size_t columnSize = vec.size() * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type);
        cudaMalloc((void**)&dst, columnSize);
        cudaMemcpy(dst, vec.data(), columnSize, cudaMemcpyHostToDevice);
        return dst;
    }
};

struct vector_to_managed_array {
    template<class T>
    T* operator() (const std::vector<T>& vec) {
        T* dst;
        size_t columnSize = vec.size() * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type);
        cudaMallocManaged((void**)&dst, columnSize);
        std::memcpy(dst, vec.data(), columnSize);
        return dst;
    }
};

template<unsigned node = 0>
struct vector_to_numa_node_array {
    template<class T>
    T* operator() (const std::vector<T>& vec) {
        size_t columnSize = vec.size() * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type);
        T* dst = reinterpret_cast<T*>(numa_alloc_onnode(columnSize, node));
        std::memcpy(dst, vec.data(), columnSize);
        return dst;
    }
};
