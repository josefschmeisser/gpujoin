#pragma once

#include <stdlib.h>
#include <new>
#include <limits>

template <class T, bool managed = false> 
struct cuda_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template <class U> struct rebind { typedef cuda_allocator<U> other; };
    cuda_allocator() throw() {}
    cuda_allocator(const cuda_allocator&) throw() {}

    template <class U> cuda_allocator(const cuda_allocator<U>&) throw(){}

    ~cuda_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const * = 0) {
        if (0 == s)
            return NULL;
        pointer temp;
        if constexpr (managed) {
            cudaMallocManaged(&temp, s * sizeof(T));
        } else {
            cudaMalloc(&temp, s * sizeof(T));
        }
        if (temp == NULL)
            throw std::bad_alloc();
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
