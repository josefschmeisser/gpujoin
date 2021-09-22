#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include <unordered_set>

#include <cuda_runtime_api.h>
#include <numa.h>

struct vector_to_device_array {
    template<class T>
    T* operator() (const std::vector<T>& vec) {
        T* dst;
        size_t columnSize = vec.size() * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type);
        cudaMalloc((void**)&dst, columnSize);
        cudaMemcpy(dst, vec.data(), columnSize, cudaMemcpyHostToDevice);
        return dst;
    }
};

struct vector_to_managed_array {
    template<class T>
    T* operator() (const std::vector<T>& vec) {
        T* dst;
        size_t columnSize = vec.size() * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type);
        cudaMallocManaged((void**)&dst, columnSize);
        std::memcpy(dst, vec.data(), columnSize);
        return dst;
    }
};

template<unsigned node = 0>
struct vector_to_numa_node_array {
    template<class T>
    T* operator() (const std::vector<T>& vec) {
        size_t columnSize = vec.size() * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type);
        T* dst = reinterpret_cast<T*>(numa_alloc_onnode(columnSize, node));
        std::memcpy(dst, vec.data(), columnSize);
        return dst;
    }
};

template<class RandomIt, class P>
std::vector<size_t> compute_permutation(RandomIt first, RandomIt last, P p) {
    std::vector<size_t> permutation(std::distance(first, last));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](const auto a, const auto b) {
        return p(*(first + a), *(first + b));
    });
    return permutation;
}

// applies a given permuation by swapping all elements along each permuation cycle
template<class PermutationVector, class... InputVectors>
void apply_permutation(PermutationVector& permutation, InputVectors&... vectors) {
    for (size_t i = 0; i < permutation.size(); ++i) {
        auto current = i;
        while (i != permutation[current]) {
            auto next = permutation[current];
            // fold expressions are a c++17 feature
            //(std::swap(vectors[current], vectors[next]), ...);
            (void) (int[]) {(std::swap(vectors[current], vectors[next]), 0)...};
            permutation[current] = current;
            current = next;
        }
        permutation[current] = current;
    }
}

template<class InputIt>
std::string stringify(InputIt first, InputIt last) {
    auto comma_fold = [](std::string a, auto b) {
        return std::move(a) + ',' + std::to_string(b);
    };
    return std::accumulate(std::next(first), last, std::to_string(*first), comma_fold);
}

// TODO test
template<class InputIt, class OutputIt, class URBG>
void simple_sample(InputIt first, InputIt last, OutputIt out, size_t n, URBG&& rg) {
    size_t size = std::distance(first, last);
    std::uniform_int_distribution<> distrib(0, size - 1);
    std::unordered_set<size_t> seen;
    while (n > 0) {
        size_t i = distrib(rg);
        if (seen.count(i) > 0) continue;
        seen.emplace(i);
        *out = std::move(*(first + i));
        ++out;
        --n;
    }
}

//template<template<class T> class Allocator>
template<class T>
struct target_memcpy {
    void* operator()(void* dest, const void* src, size_t n) {
        return memcpy(dest, src, n);
    }
};
