#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <memory>

#include "allocator_traits.hpp"
#include "cuda_allocator.hpp"

template<class T>
struct abstract_device_array {
    T* ptr_;
    size_t size_;

    abstract_device_array() : ptr_(nullptr), size_(0) {}

    abstract_device_array(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

    abstract_device_array(const abstract_device_array&) = delete;

    virtual ~abstract_device_array() = default;

    T* data() { return ptr_; }

    T* release() {
        auto tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

    size_t size() const { return size_; }

    virtual std::unique_ptr<abstract_device_array<T>> to_host_accessible() const = 0;
};

template<class Allocator>
struct managed_device_array : abstract_device_array<typename Allocator::value_type> {
    using value_type = typename Allocator::value_type;

    Allocator allocator_;

    managed_device_array(value_type* ptr, size_t size, Allocator allocator) : abstract_device_array<value_type>(ptr, size), allocator_(allocator) {}

    managed_device_array(const managed_device_array&) = delete;

    ~managed_device_array() override {
        if (this->ptr_) {
            allocator_.deallocate(this->ptr_, this->size_);
        }
    }

    std::unique_ptr<abstract_device_array<value_type>> to_host_accessible() const override;
};

template<class T>
struct unmanaged_device_array : abstract_device_array<T> {
    using value_type = T;

    unmanaged_device_array(value_type* ptr, size_t size) : abstract_device_array<value_type>(ptr, size) {}

    ~unmanaged_device_array() override = default;

    std::unique_ptr<abstract_device_array<value_type>> to_host_accessible() const override;
};

template<class Allocator>
std::unique_ptr<abstract_device_array<typename Allocator::value_type>> managed_device_array<Allocator>::to_host_accessible() const {
    if /*constexpr*/ (is_allocation_host_accessible<Allocator>::value) {
        return std::make_unique<unmanaged_device_array<value_type>>(this->ptr_, this->size_);
    } else {
        //default_cuda_allocator<T> allocator;
        std::allocator<value_type> array_allocator;
        value_type* ptr = array_allocator.allocate(this->size_);
        const auto ret = cudaMemcpy(ptr, this->ptr_, this->size_*sizeof(value_type), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return std::make_unique<managed_device_array<decltype(array_allocator)>>(ptr, this->size_, array_allocator);
    }
}

template<class T>
std::unique_ptr<abstract_device_array<T>> unmanaged_device_array<T>::to_host_accessible() const {
    //return std::make_unique<device_array<T, void>>(this->ptr_, this->size_);
    throw std::runtime_error("attempt to copy allocation of unknown origin");
}

template<class T>
struct device_array_wrapper {
    using value_type = T;

    std::unique_ptr<abstract_device_array<T>> device_array_;

    device_array_wrapper() = default;

    device_array_wrapper(const device_array_wrapper&) = delete;

    device_array_wrapper(device_array_wrapper&& other) noexcept : device_array_(other.device_array_.release()) {}

    template<class Allocator>
    device_array_wrapper(T* ptr, size_t size, Allocator allocator) {
        device_array_ = std::make_unique<managed_device_array<Allocator>>(ptr, size, allocator);
    }

    ~device_array_wrapper() = default;

private:
    device_array_wrapper(std::unique_ptr<abstract_device_array<T>>&& device_array) {
        device_array_.reset(device_array.release());
    }

public:
    device_array_wrapper& operator=(device_array_wrapper&& r) noexcept {
        device_array_.reset(r.device_array_.release());
        return *this;
    }

    template<class Allocator = cuda_allocator<T>>
    void allocate(size_t size) {
        using array_allocator_type = typename Allocator::template rebind<T>::other;
        array_allocator_type array_allocator;
        T* ptr = array_allocator.allocate(size);
        device_array_ = std::make_unique<managed_device_array<Allocator>>(ptr, size, array_allocator);
    }

    static device_array_wrapper<T> create_reference_only(T* ptr, size_t size) {
        return device_array_wrapper(std::make_unique<unmanaged_device_array<T>>(ptr, size));
    }

    T* data() {
        return (device_array_) ? device_array_->data() : nullptr;
    }

    const T* data() const {
        return (device_array_) ? device_array_->data() : nullptr;
    }

    T* release() {
        return (device_array_) ? device_array_->release() : nullptr;
    }

    void swap(device_array_wrapper& other) {
        std::swap(device_array_, other.device_array_);
    }

    size_t size() const {
        if (!device_array_) {
            throw std::runtime_error("attempt to obtain the size of an empty device_array_wrapper");
        }
        return device_array_->size();
    }

    device_array_wrapper<T> to_host_accessible() const {
        if (device_array_) {
            return device_array_wrapper(device_array_->to_host_accessible());
        }
        return device_array_wrapper();
    }
};

template<class T>
auto create_device_array(size_t size) {
    using array_allocator_type = cuda_allocator<T>;
    array_allocator_type array_allocator;
    T* ptr = array_allocator.allocate(size);
    return device_array_wrapper<T>(ptr, size, array_allocator);
}

template<class T, class Allocator>
auto create_device_array(size_t size, Allocator& allocator) {
    using array_allocator_type = typename Allocator::template rebind<T>::other;
    array_allocator_type array_allocator = allocator;
    T* ptr = array_allocator.allocate(size);
    return device_array_wrapper<T>(ptr, size, array_allocator);
}

template<typename T, class InputAllocator, typename OutputAllocator, template <typename, typename> class VectorType>
auto create_device_array_from(VectorType<T, InputAllocator>& vec, OutputAllocator& device_allocator) {
    using array_allocator_type = typename OutputAllocator::template rebind<T>::other;

    // check if rebinding is sufficient
    if (std::is_same<InputAllocator, array_allocator_type>::value) {
        printf("same allocator after all\n");
        return device_array_wrapper<T>::create_reference_only(vec.data(), vec.size());
    }

    // allocate memory
    array_allocator_type array_allocator;
    T* ptr = array_allocator.allocate(vec.size()); // allocators already take the target type size into account

    // we are limited to c++14, so no constexpr if here...
    if (is_cuda_allocator<OutputAllocator>::value) {
        // we have to use cudaMemcpy here since device memory can't be accessed by the host
        const auto ret = cudaMemcpy(ptr, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return device_array_wrapper<T>(ptr, vec.size(), array_allocator);
    } else {
        std::memcpy(ptr, vec.data(), vec.size()*sizeof(T));
        return device_array_wrapper<T>(ptr, vec.size(), array_allocator);
    }
}

template<typename T, class InputAllocator, typename OutputAllocator, template <typename, typename> class VectorType>
auto create_device_array_from(VectorType<T, OutputAllocator>& vec, OutputAllocator& device_allocator) {
    return device_array_wrapper<T>::create_reference_only(vec.data(), vec.size());
}

// This alias template removes the depency on the additional template parameter of cuda_allocator
template<class T>
using default_cuda_allocator = cuda_allocator<T>;

template<class T, template<class U> class OutputAllocator = default_cuda_allocator>
auto create_device_array_from(const T* arr, size_t size) {
    static OutputAllocator<T> allocator;
    T* ptr = allocator.allocate(size);
    target_memcpy<OutputAllocator<T>>()(ptr, arr, size*sizeof(T));
    return device_array_wrapper<T>(ptr, size, allocator);
}
