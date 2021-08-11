#pragma once

#include <stdlib.h>
#include <stdexcept>
#include <limits>
#include <string>

#include <sys/mman.h>
#include <linux/mman.h>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>

#include <sys/mman.h>
#include <linux/mman.h>

template <class T> 
struct huge_page_allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    unsigned default_node_ = 0;
    enum page_type : unsigned {
        transparent_huge = 0,
        huge_2mb = 21,
        huge_16mb = 24,
        huge_1gb = 30,
        huge_16gb = 34
    } default_page_type_;

    template <class U> struct rebind { typedef huge_page_allocator<U> other; };
    huge_page_allocator() throw() : default_node_(0), default_page_type_(transparent_huge) {}
    huge_page_allocator(unsigned default_node, page_type default_page_type) throw() : default_node_(default_node), default_page_type_(default_page_type) {}
    huge_page_allocator(const huge_page_allocator& other) throw() : default_node_(other.default_node_), default_page_type_(other.default_page_type_) {}

    template <class U> huge_page_allocator(const huge_page_allocator<U>& other) throw() : default_node_(other.default_node_), default_page_type_(other.default_page_type_) {}

    ~huge_page_allocator() throw() {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    size_type round_to_next_page(size_type size, size_type page_size) {
        const auto align_mask = ~(page_size - 1);
        return (size + page_size - 1) & align_mask;
    }

    pointer allocate(size_type s, void const * = 0) {
        using namespace std::string_literals;

        if (0 == s) {
            return nullptr;
        }

        if (s > max_size()) {
            throw std::bad_array_new_length();
        }

        int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;

        // force huge page size
        mmap_flags |= (default_page_type_ == huge_2mb)  ? MAP_HUGE_2MB  : 0;
        mmap_flags |= (default_page_type_ == huge_16mb) ? MAP_HUGE_16MB : 0;
        mmap_flags |= (default_page_type_ == huge_1gb)  ? MAP_HUGE_1GB  : 0;
        mmap_flags |= (default_page_type_ == huge_16gb) ? MAP_HUGE_16GB : 0;

        const size_type size = s*sizeof(T);
        void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
        if (!ptr) {
            throw std::runtime_error("mmap failed: "s + std::strerror(errno));
        }

        if (default_page_type_ == transparent_huge) {
            const auto r = madvise(ptr, size, MADV_HUGEPAGE);
            if (r != 0) {
                throw std::runtime_error("madvise failed: "s + std::strerror(errno));
            }
        }

        const auto aligned_size = round_to_next_page(size, 1 << default_page_type_);
        const unsigned long allowed_nodes = *numa_get_mems_allowed()->maskp;
        const unsigned long maxnode = numa_get_mems_allowed()->size;
        const unsigned long node_mask = (1 << default_node_) & allowed_nodes;
        if (node_mask == 0) {
            throw std::runtime_error("node not available for process");
        }

        const auto r = mbind(ptr, aligned_size, MPOL_BIND, &node_mask, maxnode, MPOL_MF_STRICT);
        if (r != 0) {
            throw std::runtime_error("mbind failed: "s + std::strerror(errno));
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type s) {
        using namespace std::string_literals;

        const size_type size = s*sizeof(T);
        const auto aligned_size = round_to_next_page(size, 1 << default_page_type_);
        const auto r = munmap(p, aligned_size);
        if (r != 0) {
            throw std::runtime_error("munmap failed: "s + std::strerror(errno));
        }
    }

    size_type max_size() const throw() {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    void construct(pointer p, const T& val) {
        new((void *)p) T(val);
    }

    void destroy(pointer p) {
        p->~T();
    }

    unsigned default_node() const {
        return default_node_;
    }
};
