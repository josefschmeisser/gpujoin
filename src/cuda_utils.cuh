#pragma once

#include <cstdint>

#define FULL_MASK 0xffffffff

#ifndef GPU_CACHE_LINE_SIZE
#define GPU_CACHE_LINE_SIZE 128
#endif

template<class CharType>
__device__ int device_strcmp(const CharType* str_a, const CharType* str_b, unsigned len) {
    int match = 0;
    unsigned i = 0;
    unsigned done = 0;
    while ((i < len) && (match == 0) && !done) {
        if ((str_a[i] == 0) || (str_b[i] == 0)) {
            done = 1;
        } else if (str_a[i] != str_b[i]) {
            match = i+1;
            if (((int)str_a[i] - (int)str_b[i]) < 0) match = 0 - (i + 1);
        }
        i++;
    }
    return match;
}

// use with care, see: https://stackoverflow.com/a/4433731
__forceinline__ __device__ unsigned lane_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

template<class T>
__device__ T atomic_add_sat(T* address, T val, T saturation) {
    unsigned expected, update, old;
    old = *address;
    do {
        expected = old;
        update = (old + val > saturation) ? saturation : old + val;
        old = atomicCAS(address, expected, update);
    } while (expected != old);
    return old;
}

template<class T>
__device__ T atomic_sub_safe(T* address, T val) {
    unsigned expected, update, old;
    old = *address;
    do {
        expected = old;
        update = (old > val) ? (old - val) : 0;
        old = atomicCAS(address, expected, update);
    } while (expected != old);
    return old;
}

template<class T>
__forceinline__ __device__ T round_up_pow2(T value) {
    return static_cast<T>(1) << (sizeof(T)*8 - __clz(value - 1));
}

// 32bit Murmur3 hash
template <
    class T,
    std::enable_if_t<std::is_integral<T>::value && sizeof(T) == sizeof(uint32_t), int> = 0
>
__host__ __device__ uint32_t murmur3_hash(const T k) {
    uint32_t ik = static_cast<uint32_t>(k);

    ik ^= ik >> 16;
    ik *= 0x85ebca6b;
    ik ^= ik >> 13;
    ik *= 0xc2b2ae35;
    ik ^= ik >> 16;
    return ik;
}

//    __device__ uint64_t hash(uint64_t k)
//template<class T>
//__device__ std::enable_if_t<std::is_integral<T>::value && sizeof(T) == sizeof(uint64_t), T> hash(const T k) {

// 64bit Murmur3 hash
template <
    class T,
    std::enable_if_t<std::is_integral<T>::value && sizeof(T) == sizeof(uint64_t), int> = 0
>
__host__ __device__ uint64_t murmur3_hash(const T k) {
    uint64_t ik = static_cast<uint64_t>(k);

    ik ^= ik >> 33;
    ik *= 0xff51afd7ed558ccd;
    ik ^= ik >> 33;
    ik *= 0xc4ceb9fe1a85ec53;
    ik ^= ik >> 33;

    return ik;
}

// maps integral integer types to their respective atomicCAS input types
template <class T, class T2 = void>
struct to_cuda_atomic_input;

// 16bit integral types
template <class T>
struct to_cuda_atomic_input<
    T,
    typename std::enable_if<std::is_integral<T>::value && sizeof(T) == sizeof(uint16_t)>::type
> {
    using type = unsigned short int;
};

// 32bit integral types
template <class T>
struct to_cuda_atomic_input<
    T,
    typename std::enable_if<std::is_integral<T>::value && sizeof(T) == sizeof(uint32_t)>::type
> {
    using type = unsigned int;
};

// 64bit integral types
template <class T>
struct to_cuda_atomic_input<
    T,
    typename std::enable_if<std::is_integral<T>::value && sizeof(T) == sizeof(uint64_t)>::type
> {
    using type = unsigned long long int;
};

template <class T>
__device__ T tmpl_atomic_add(T* address, T val) {
    using input_type = typename to_cuda_atomic_input<T>::type;
    static_assert(sizeof(input_type) == sizeof(T));
    return atomicAdd(
        reinterpret_cast<input_type*>(address),
        static_cast<input_type>(val)
    );
}

template <class T>
__device__ T tmpl_atomic_cas(T* address, T compare, T val) {
    using input_type = typename to_cuda_atomic_input<T>::type;
    static_assert(sizeof(input_type) == sizeof(T));
    return atomicCAS(
        reinterpret_cast<input_type*>(address),
        static_cast<input_type>(compare),
        static_cast<input_type>(val)
    );
}
