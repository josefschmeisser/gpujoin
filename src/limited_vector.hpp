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
        : arr_(nullptr), limit_(0), size_(0) {}

    limited_vector(size_t limit, size_t count = 0)
        : limit_(limit), size_(count)
    {
        static Allocator allocator;
        arr_ = allocator.allocate(limit);
    }

    ~limited_vector() {
        static Allocator allocator;
        if (arr_) {
            allocator.deallocate(arr_, limit_);
        }
    }

    void swap(my_type& other) noexcept {
        std::swap(arr_, other.arr_);
        std::swap(size_, other.size_);
        std::swap(limit_, other.limit_);
    }

    template<class... Args>
    void emplace_back(Args&&... args) {
        if (size_ + 1 > limit_) {
            throw std::runtime_error("limited_vector capacity exceeded");
        }

        new (&arr_[size_++]) T(args...);
        assert(size_ <= limit_);
    }

    T& front() noexcept {
        assert(size_ > 0);
        return arr_[0];
    }

    const T& front() const noexcept {
        assert(size_ > 0);
        return arr_[0];
    }

    T& back() noexcept {
        assert(size_ > 0);
        return arr_[size_ - 1];
    }

    const T& back() const noexcept {
        assert(size_ > 0);
        return arr_[size_ - 1];
    }

    size_t size() const noexcept { return size_; }

    size_t capacity() const noexcept { return limit_; }

    T& operator[](size_t idx) noexcept {
        if (idx >= size_) printf("idx: %lu > size: %lu\n", idx, size_);
        assert(idx < size_);
        return arr_[idx];
    }

    const T& operator[](size_t idx) const noexcept {
        assert(idx < size_);
        return arr_[idx];
    }

    auto begin() noexcept { return arr_; }

    const auto begin() const noexcept { return arr_; }

    auto end() noexcept { return arr_ + size_; }

    const auto end() const noexcept { return arr_ + size_; }

    T* data() noexcept { return arr_; }

    const T* data() const noexcept { return arr_; }

private:
    T* arr_;
    size_t size_;
    size_t limit_;
};
