#pragma once

#include <cassert>
#include <cstdint>
#include <numeric>
#include <memory>
#include <unordered_set>
#include <vector>

#include "utils.hpp"
#include "zipf.hpp"

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"

enum class dataset_type : unsigned { dense, uniform };

enum class lookup_pattern_type : unsigned { uniform, zipf };

template<class KeyType, class IndexStructureType, class VectorType>
void generate_datasets(dataset_type dt, unsigned max_bits, VectorType& keys, lookup_pattern_type lookup_pattern, double zipf_factor, VectorType& lookups) {
    auto rng = std::default_random_engine {};

    if (dt == dataset_type::dense) {
        std::iota(keys.begin(), keys.end(), 0);
    } else if (dt == dataset_type::uniform) {
        // create random keys
        std::uniform_int_distribution<> key_distrib(0, 1 << (max_bits - 1));
        std::unordered_set<KeyType> unique;
        unique.reserve(keys.size());
        while (unique.size() < keys.size()) {
            const auto key = key_distrib(rng);
            unique.insert(key);
        }

        std::copy(unique.begin(), unique.end(), keys.begin());
        std::sort(keys.begin(), keys.end());
    } else {
        assert(false);
    }

    if (lookup_pattern == lookup_pattern_type::uniform) {
        std::uniform_int_distribution<> lookup_distribution(0, keys.size() - 1);
        std::generate(lookups.begin(), lookups.end(), [&]() { return keys[lookup_distribution(rng)]; });
    } else if (lookup_pattern == lookup_pattern_type::zipf) {
        std::mt19937 generator;
        generator.seed(0);
        zipf_distribution<uint64_t> lookup_distribution(keys.size(), zipf_factor);
        for (uint64_t i = 0; i < lookups.size(); ++i) {
            const auto key_pos = lookup_distribution(generator);
            lookups[i] = keys[key_pos];
        }
    } else {
        assert(false);
    }

    //std::sort(lookups.begin(), lookups.end());
}

template<class KeyType, class IndexStructureType, class VectorType>
std::unique_ptr<IndexStructureType> build_index(const VectorType& h_keys, KeyType* d_keys) {
    auto index = std::make_unique<IndexStructureType>();
    index->construct(h_keys, d_keys);
    printf("index size: %lu bytes\n", index->memory_consumption());
    return index;
}
