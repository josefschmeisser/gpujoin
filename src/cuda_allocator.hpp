#pragma once

#include <stdlib.h>
#include <cstddef>
#include <new>
#include <limits>

#include <cuda_runtime.h>

#include "allocator_traits.hpp"
#include "utils.hpp"

enum class cuda_allocation_type : unsigned {
    device,
    unified,
    zero_copy
};

template <class T, cuda_allocation_type CudaAllocationType = cuda_allocation_type::device> 
struct cuda_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template <class U> struct rebind { typedef cuda_allocator<U, CudaAllocationType> other; };
    cuda_allocator() throw() {}
    cuda_allocator(const cuda_allocator&) throw() {}

    template <class U, cuda_allocation_type OtherCudaAllocationType> cuda_allocator(const cuda_allocator<U, OtherCudaAllocationType>&) throw(){}

    ~cuda_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const * = 0) {
        if (0 == s) {
            return NULL;
        }

        if (s > max_size()) {
            throw std::bad_array_new_length();
        }

        pointer temp = nullptr;
        if /*constexpr*/ (CudaAllocationType == cuda_allocation_type::device) {
            cudaMalloc(&temp, s * sizeof(T));
        } else if (CudaAllocationType == cuda_allocation_type::unified) {
            //printf("allocate managed\n");
            cudaMallocManaged(&temp, s * sizeof(T));
            //__host__ ​cudaError_t cudaMemAdvise ( const void* devPtr, size_t count, cudaMemoryAdvise advice, int  device ) 
            cudaMemAdvise(temp, s * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        } else {
            // zero copy
            pointer host_ptr;
            cudaMallocHost(&host_ptr, s * sizeof(T));
            cudaHostGetDevicePointer(&temp, host_ptr, 0);
            printf("zero copy\n");
        }
        if (temp == nullptr) {
            throw std::bad_alloc();
        }

        return temp;
    }

    void deallocate(pointer p, size_type) {
        cudaFree(p);
    }

    size_type max_size() const throw() {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    void construct(pointer p, const T& val) {
        new((void *)p) T(val);
    }

    void destroy(pointer p) {
        p->~T();
    }
};

template<class T>
struct target_memcpy<cuda_allocator<T, cuda_allocation_type::device>> {
    void* operator()(void* dest, const void* src, size_t n) {
        const auto ret = cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return dest;
    }
};

template<class T, cuda_allocation_type CudaAllocationType>
struct is_cuda_allocator<cuda_allocator<T, CudaAllocationType>> {
    static constexpr bool value = true;
};

template<class T>
struct is_allocation_host_accessible<cuda_allocator<T, cuda_allocation_type::device>> {
    static constexpr bool value = false;
};
