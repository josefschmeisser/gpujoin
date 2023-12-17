#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template<class T>
struct vector_view {
    using my_type = vector_view<T>;
    using value_type = T;
    using size_type = size_t;

    vector_view() noexcept
        : arr_(nullptr), size_(0) {}

    vector_view(T* arr, size_type size) noexcept
        : arr_(arr), size_(size)
    {}

    vector_view(vector_view&& other) noexcept {
        arr_ = other.arr_;
        size_ = other.size_;
        other.arr_ = nullptr;
        other.size_ = 0;
    }

    ~vector_view() = default;

    vector_view& operator=(vector_view&& other) {
        arr_ = other.arr_;
        size_ = other.size_;
        other.arr_ = nullptr;
        other.size_ = 0;
        return *this;
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

    size_type size() const noexcept { return size_; }

    size_type capacity() const noexcept { return size_; }

    T& operator[](size_t idx) noexcept {
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
};

// for pre-c++17 compilers
template<class T>
auto make_vector_view(T* arr, size_t size) {
    return vector_view<T>(arr, size);
}

template<class VectorType>
vector_view<typename VectorType::value_type> make_vector_view(VectorType& vector_value) {
    return vector_view<typename VectorType::value_type>(&vector_value.front(), vector_value.size());
}

template<class VectorType>
vector_view<const typename VectorType::value_type> make_vector_view(const VectorType& vector_value) {
    return vector_view<const typename VectorType::value_type>(&vector_value.front(), vector_value.size());
}
