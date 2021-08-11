#pragma once

#include <stdlib.h>
#include <cstddef>
#include <new>
#include <limits>

#include <cuda_runtime.h>

#include "allocator_traits.hpp"
#include "utils.hpp"

template <class T, bool Managed = false> 
struct cuda_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template <class U> struct rebind { typedef cuda_allocator<U, Managed> other; };
    cuda_allocator() throw() {}
    cuda_allocator(const cuda_allocator&) throw() {}

    template <class U, bool OtherManaged> cuda_allocator(const cuda_allocator<U, OtherManaged>&) throw(){}

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
        if /*constexpr*/ (Managed) {
            cudaMallocManaged(&temp, s * sizeof(T));
        } else {
            cudaMalloc(&temp, s * sizeof(T));
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
struct target_memcpy<cuda_allocator<T, false>> {
    void* operator()(void* dest, void* src, size_t n) {
        const auto ret = cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return dest;
    }
};

template<class T, bool Managed>
struct is_cuda_allocator<cuda_allocator<T, Managed>> {
    static constexpr bool value = true;
};

template<class T>
struct is_allocation_host_accessible<cuda_allocator<T, false>> {
    static constexpr bool value = false;
};
