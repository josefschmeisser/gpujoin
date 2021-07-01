#pragma once

#include <stdexcept>
#include <string>
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"

template<class T>
struct abstract_device_array {
    T* ptr_;
    size_t size_;

    abstract_device_array() : ptr_(nullptr), size_(0) {}

    abstract_device_array(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

    abstract_device_array(const abstract_device_array&) = delete;

    T* data() { return ptr_; }
};

template<class T, class Allocator>
struct device_array : abstract_device_array<T> {
    Allocator allocator_;

    device_array(T* ptr, size_t size, Allocator allocator) : abstract_device_array<T>(ptr, size), allocator_(allocator) {}

    device_array(const device_array&) = delete;

    ~device_array() {
        allocator_.deallocate(this->ptr_, sizeof(T)*this->size_);
    }
};

template<class T>
struct device_array<T, void> : abstract_device_array<T> {
    device_array(T* ptr, size_t size) : abstract_device_array<T>(ptr, size) {}
};

template<class T>
struct device_array_wrapper {
    std::unique_ptr<abstract_device_array<T>> device_array_;

    device_array_wrapper() = default;

    template<class Allocator>
    device_array_wrapper(T* ptr, size_t size, Allocator allocator) {
        device_array_ = std::make_unique<device_array<T, Allocator>>(ptr, size, allocator);
    }

    device_array_wrapper(T* ptr, size_t size) {
        device_array_ = std::make_unique<device_array<T, void>>(ptr, size);
    }

    T* data() { return device_array_->data(); }
};

template<class T, class OutputAllocator, class InputAllocator>
auto create_device_array_from(std::vector<T, InputAllocator>& vec, OutputAllocator& allocator) {
    printf("different types\n");
    if constexpr (std::is_same<OutputAllocator, numa_allocator<T>>::value) {
        T* ptr = allocator.allocate(vec.size()*sizeof(T));
        std::memcpy(ptr, vec.data(), vec.size()*sizeof(T));
        return device_array_wrapper(ptr, vec.size(), allocator);
    } else if (std::is_same<OutputAllocator, cuda_allocator<T, true>>::value) {
        return device_array_wrapper(vec.data(), vec.size());
    } else if (std::is_same<OutputAllocator, cuda_allocator<T, false>>::value) {
        T* ptr;
        auto ret = cudaMalloc((void**)&ptr, vec.size()*sizeof(T));
        if (ret != cudaSuccess) throw std::runtime_error("cudaMalloc failed, code: " + std::to_string(ret));
        ret = cudaMemcpy(ptr, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMalloc failed, code: " + std::to_string(ret));
        return device_array_wrapper(ptr, vec.size(), allocator);
    }
    throw std::runtime_error("not available");
}

template<class T, class OutputAllocator>
auto create_device_array_from(std::vector<T, OutputAllocator>& vec, OutputAllocator& allocator) {
    printf("same type\n");
    if constexpr (std::is_same<OutputAllocator, numa_allocator<T>>::value) {
        if (allocator.node() == vec.get_allocator().node()) {
            return device_array_wrapper(vec.data(), vec.size());
        } else {
            T* ptr = allocator.allocate(vec.size()*sizeof(T));
            std::memcpy(ptr, vec.data(), vec.size()*sizeof(T));
            return device_array_wrapper(ptr, vec.size(), allocator);
        }
    } else if (std::is_same<OutputAllocator, cuda_allocator<T, false>>::value) {
        return device_array_wrapper(vec.data(), vec.size());
    }
    throw std::runtime_error("not available");
}
