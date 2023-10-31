#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <memory>
#include <type_traits>

#include "allocator_traits.hpp"
#include "cuda_allocator.hpp"

template<class T>
struct allocation_concept {
    T* ptr_;
    size_t size_;

    allocation_concept() : ptr_(nullptr), size_(0) {}

    allocation_concept(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

    allocation_concept(const allocation_concept&) = delete;

    virtual ~allocation_concept() = default;

    T* data() { return ptr_; }

    T* release() {
        auto tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

    size_t size() const { return size_; }

    virtual bool is_device_accessible() const = 0;

    virtual bool is_host_accessible() const = 0;

    virtual std::shared_ptr<allocation_concept<T>> to_host_accessible() const = 0;
};

template<class Allocator>
struct allocation_model : public allocation_concept<typename Allocator::value_type> {
    using value_type = typename Allocator::value_type;

    Allocator allocator_;
    bool managed_;

    allocation_model(value_type* ptr, size_t size, Allocator allocator, bool managed = true)
        : allocation_concept<value_type>(ptr, size)
        , allocator_(allocator)
        , managed_(managed)
    {}

    allocation_model(const allocation_model&) = delete;

    ~allocation_model() override {
        if (managed_ && this->ptr_) {
            allocator_.deallocate(this->ptr_, this->size_);
        }
    }

    bool is_device_accessible() const override;

    bool is_host_accessible() const override;

    std::shared_ptr<allocation_concept<value_type>> to_host_accessible() const override;

    template<class DestAllocator>
    std::shared_ptr<allocation_concept<value_type>> copy_if_different(DestAllocator& allocator) const;
};

template<class Allocator>
bool allocation_model<Allocator>::is_device_accessible() const {
    throw "not implemented";
    //return is_allocation_device_accessible<Allocator>::value;
}

template<class Allocator>
bool allocation_model<Allocator>::is_host_accessible() const {
    return is_allocation_host_accessible<Allocator>::value;
}

template<class Allocator>
std::shared_ptr<allocation_concept<typename Allocator::value_type>> allocation_model<Allocator>::to_host_accessible() const {
    static std::allocator<value_type> target_allocator;

    if /*constexpr*/ (is_allocation_host_accessible<Allocator>::value) {
        return std::make_shared<allocation_model<decltype(allocator_)>>(this->ptr_, this->size_, allocator_, false);
    } else {
        value_type* ptr = target_allocator.allocate(this->size_);
        const auto ret = cudaMemcpy(ptr, this->ptr_, this->size_*sizeof(value_type), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return std::make_shared<allocation_model<decltype(target_allocator)>>(ptr, this->size_, target_allocator);
    }
}

template<class Allocator>
template<class DestAllocator>
std::shared_ptr<allocation_concept<typename Allocator::value_type>> allocation_model<Allocator>::copy_if_different(DestAllocator& allocator) const {
#if 0
    using src_allocator_type = typename Allocator::template rebind<value_type>::other;
    using dst_allocator_type = typename DestAllocator::template rebind<value_type>::other;

    if /*constexpr*/ (std::is_same<src_allocator_type, dst_allocator_type>::value) {
        return std::make_shared<allocation_model<dst_allocator_type>>(this->ptr_, this->size_, allocator_, false);
    } else {
        const bool src_host_accessible = is_allocation_host_accessible<src_allocator_type>::value;
        const bool dst_host_accessible = is_allocation_host_accessible<dst_allocator_type>::value;

        //cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice
 	    enum cudaMemcpyKind kind = cudaMemcpyHostToHost;
        kind = (src_host_accessible && dst_host_accessible) ? cudaMemcpyHostToHost : kind;
        kind = (src_host_accessible && !dst_host_accessible) ? cudaMemcpyHostToDevice : kind;
        kind = (!src_host_accessible && dst_host_accessible) ? cudaMemcpyDeviceToHost : kind;
        kind = (!src_host_accessible && !dst_host_accessible) ? cudaMemcpyDeviceToDevice : kind;

        dst_allocator_type bound_allocator = allocator;
        value_type* ptr = bound_allocator.allocate(this->size_);

        const auto ret = cudaMemcpy(ptr, this->ptr_, this->size_*sizeof(value_type), kind);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return std::make_shared<allocation_model<dst_allocator_type>>(ptr, this->size_, bound_allocator, true);
    }
#endif
    return {};
}

template<class T>
struct device_array_wrapper {
    using value_type = T;

    std::shared_ptr<allocation_concept<T>> device_array_;

    device_array_wrapper() = default;

    device_array_wrapper(const device_array_wrapper& other) noexcept
        : device_array_(other.device_array_)
    {}

    template<class Allocator>
    device_array_wrapper(T* ptr, size_t size, Allocator allocator) {
        using bound_allocator_type = typename Allocator::template rebind<T>::other;
        static bound_allocator_type bound_allocator;
        device_array_ = std::make_shared<allocation_model<bound_allocator_type>>(ptr, size, bound_allocator, true);
    }

    ~device_array_wrapper() = default;

private:
    device_array_wrapper(std::shared_ptr<allocation_concept<T>>&& device_array) {
        device_array_ = device_array;
    }

public:
    device_array_wrapper& operator=(const device_array_wrapper& r) noexcept {
        device_array_ = r.device_array_;
        return *this;
    }

    template<class Allocator = cuda_allocator<T>>
    void allocate(size_t size) {
        using bound_allocator_type = typename Allocator::template rebind<T>::other;
        static bound_allocator_type bound_allocator;
        T* ptr = bound_allocator.allocate(size);
        device_array_ = std::make_shared<allocation_model<bound_allocator_type>>(ptr, size, bound_allocator, true);
    }

    template<class Allocator>
    static device_array_wrapper<T> create_reference_only(T* ptr, size_t size, Allocator allocator) {
        using bound_allocator_type = typename Allocator::template rebind<T>::other;
        static bound_allocator_type bound_allocator;
        return device_array_wrapper(std::make_shared<allocation_model<bound_allocator_type>>(ptr, size, bound_allocator, false));
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

    bool is_host_accessible() const {
        if (device_array_) {
            return device_array_->is_host_accessible();
        }
        return false;
    }

    device_array_wrapper<T> to_host_accessible() const {
        if (device_array_) {
            return device_array_wrapper(device_array_->to_host_accessible());
        }
        return device_array_wrapper();
    }

    template<class HostAllocator>
    device_array_wrapper<T> to_host_accessible(HostAllocator allocator) const {
        if (!device_array_) return device_array_wrapper();

        if (is_host_accessible()) return *this;

        using bound_allocator_type = typename HostAllocator::template rebind<T>::other;
        bound_allocator_type bound_allocator = allocator;

        device_array_wrapper<T> new_allocation;
        new_allocation.allocate(size());
        const auto ret = cudaMemcpy(new_allocation.data(), data(), size()*sizeof(T), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));

        return new_allocation;
    }

    template<class Allocator>
    device_array_wrapper<T> copy_if_different(Allocator& allocator) const {
        throw "not implemented";
    }
};

template<class T>
auto create_device_array(size_t size) {
    using allocator_type = cuda_allocator<T>;
    static allocator_type allocator;
    T* ptr = allocator.allocate(size);
    return device_array_wrapper<T>(ptr, size, allocator);
}

template<class T, class Allocator>
auto create_device_array(size_t size, Allocator& allocator) {
    using bound_allocator_type = typename Allocator::template rebind<T>::other;
    bound_allocator_type bound_allocator = allocator;
    T* ptr = bound_allocator.allocate(size);
    return device_array_wrapper<T>(ptr, size, bound_allocator);
}

template<typename T, class InputAllocator, typename OutputAllocator, template <typename, typename> class VectorType>
auto create_device_array_from(VectorType<T, InputAllocator>& vec, OutputAllocator& device_allocator) {
    using src_allocator_type = typename InputAllocator::template rebind<T>::other;
    using dst_allocator_type = typename OutputAllocator::template rebind<T>::other;
    static dst_allocator_type bound_allocator = device_allocator;

    // check if rebinding is sufficient
    if (std::is_same<src_allocator_type, dst_allocator_type>::value) {
        printf("same allocator after all\n");
        return device_array_wrapper<T>::create_reference_only(vec.data(), vec.size(), bound_allocator);
    }

    // allocate memory
    T* ptr = bound_allocator.allocate(vec.size()); // allocators already take the target type size into account

    // we are limited to c++14, so no constexpr if here...
    if (is_cuda_allocator<OutputAllocator>::value) {
        // we have to use cudaMemcpy here since device memory can't be accessed by the host
        const auto ret = cudaMemcpy(ptr, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed, code: " + std::to_string(ret));
        return device_array_wrapper<T>(ptr, vec.size(), bound_allocator);
    } else {
        std::memcpy(ptr, vec.data(), vec.size()*sizeof(T));
        return device_array_wrapper<T>(ptr, vec.size(), bound_allocator);
    }
}

template<typename T, class InputAllocator, typename OutputAllocator, template <typename, typename> class VectorType>
auto create_device_array_from(VectorType<T, OutputAllocator>& vec, OutputAllocator& device_allocator) {
    using bound_allocator_type = typename OutputAllocator::template rebind<T>::other;
    static bound_allocator_type bound_allocator = device_allocator;
    return device_array_wrapper<T>::create_reference_only(vec.data(), vec.size(), bound_allocator);
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
