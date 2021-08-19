#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template<class T, class Allocator>
struct vector_view {
    using my_type = vector_view<T, Allocator>;
    using value_type = T;

    vector_view()
        : arr_(nullptr), size_(0) {}

    vector_view(const T* arr, size_t size)
        : arr_(arr), size_(size)
    {}

    ~vector_view() = default;

    T& back() noexcept { return arr_[size_ - 1]; }

    const T& back() const noexcept { return arr_[size_ - 1]; }

    size_t size() const noexcept { return size_; }

    size_t capacity() const noexcept { return size_; }

    T& operator[](int idx) noexcept { return arr_[idx]; }

    const T& operator[](int idx) const noexcept { return arr_[idx]; }

    auto begin() noexcept { return arr_; }

    auto end() noexcept { return arr_ + size_; }

private:
    const T* arr_;
    const size_t size_;
};
