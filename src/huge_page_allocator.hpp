#pragma once

#include <stdlib.h>
#include <stdexcept>
#include <limits>

#include <sys/mman.h>
#include <linux/mman.h>
#include <numa.h>

template <class T> 
struct hage_page_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    unsigned default_node_ = 0;

    template <class U> struct rebind { typedef hage_page_allocator<U> other; };
    hage_page_allocator(unsigned node) throw() : node_(node) {}
    hage_page_allocator(const hage_page_allocator& other) throw() : node_(other.node_) {}

    template <class U> hage_page_allocator(const hage_page_allocator<U>& other) throw() : node_(other.node_) {}

    ~hage_page_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const * = 0) {
        // TODO
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
