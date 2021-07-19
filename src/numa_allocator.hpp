#pragma once

#include <stdlib.h>
#include <new>
#include <limits>

#include <numa.h>

template <class T> 
struct numa_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    unsigned node_ = 0;

    template <class U> struct rebind { typedef numa_allocator<U> other; };
    numa_allocator(unsigned node) throw() : node_(node) {}
    numa_allocator(const numa_allocator& other) throw() : node_(other.node_) {}

    template <class U> numa_allocator(const numa_allocator<U>& other) throw() : node_(other.node_) {}

    ~numa_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const * = 0) {
        if (0 == s) {
            return NULL;
        }
        pointer temp = numa_alloc_onnode(s * sizeof(T), node_);
        if (temp == NULL) {
            throw std::bad_alloc();
        }
        return temp;
    }

    void deallocate(pointer p, size_type s) {
        numa_free(p, s);
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

    unsigned node() const {
        return node_;
    }
};
