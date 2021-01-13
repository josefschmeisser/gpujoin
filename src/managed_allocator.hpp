#pragma once

#include <stdlib.h>
#include <new>
#include <limits>

template <class T> 
struct managed_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template <class U> struct rebind { typedef managed_allocator<U> other; };
    managed_allocator() throw() {}
    managed_allocator(const managed_allocator&) throw() {}

    template <class U> managed_allocator(const managed_allocator<U>&) throw(){}

    ~managed_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const * = 0) {
        if (0 == s)
            return NULL;
        pointer temp;
        cudaMallocManaged(&temp, s * sizeof(T));
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
