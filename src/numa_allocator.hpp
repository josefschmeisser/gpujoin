#pragma once

#include <stdlib.h>
#include <new>
#include <limits>

#include <numa.h>

template <class T, unsigned NumaNode> 
struct numa_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    //unsigned node_ = 0;
/*
    template <class U> struct rebind { typedef numa_allocator<U> other; };
    numa_allocator(unsigned node) throw() : node_(node) {}
    numa_allocator(const numa_allocator& other) throw() : node_(other.node_) {}

    template <class U> numa_allocator(const numa_allocator<U>& other) throw() : node_(other.node_) {}
*/
    template <class U> struct rebind { typedef numa_allocator<U, NumaNode> other; };
    numa_allocator() throw() {}
    numa_allocator(const numa_allocator& other) throw() {}

    template <class U> numa_allocator(const numa_allocator<U, NumaNode>& other) throw() {}

    ~numa_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const * = 0) {
        if (0 == s) {
            return NULL;
        }

        if (s > max_size()) {
            throw std::bad_array_new_length();
        }

        pointer temp = numa_alloc_onnode(s * sizeof(T), NumaNode);
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
/*
    unsigned node() const {
        return node_;
    }*/
};

template<class T>
struct is_numa_allocator {
    static constexpr bool value = false;
};

template<class T, unsigned NumaNode>
struct is_numa_allocator<numa_allocator<T, NumaNode>> {
    static constexpr bool value = true;
};
