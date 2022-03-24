#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

template<class T, class Allocator>
struct limited_vector {
    using my_type = limited_vector<T, Allocator>;
    using value_type = T;

    limited_vector() noexcept
        : vec_(nullptr), limit_(0), size_(0) {}

    limited_vector(size_t limit)
        : limit_(limit), size_(0)
    {
        static Allocator allocator;
        vec_ = allocator.allocate(limit);
    }

    ~limited_vector() {
        static Allocator allocator;
        if (vec_) {
            allocator.deallocate(vec_, limit_);
        }
    }

    void swap(my_type& other) noexcept {
        std::swap(vec_, other.vec_);
        std::swap(size_, other.size_);
        std::swap(limit_, other.limit_);
    }

    template<class... Args>
    void emplace_back(Args&&... args) {
        if (size_ + 1 > limit_) {
            throw std::runtime_error("limited_vector capacity exceeded");
        }

        new (&vec_[size_++]) T(args...);
        assert(size_ <= limit_);
    }

    T& back() noexcept { return vec_[size_ - 1]; }

    const T& back() const noexcept { return vec_[size_ - 1]; }

    size_t size() const noexcept { return size_; }

    size_t capacity() const noexcept { return limit_; }

private:
    T* vec_;
    size_t size_;
    size_t limit_;
};
