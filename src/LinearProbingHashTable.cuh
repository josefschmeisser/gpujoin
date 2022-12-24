#pragma once

#include <limits>

#include "cuda_utils.cuh"
#include "device_definitions.hpp"

// simple linear probing hash table
// implementation is based on: https://github.com/nosferalatu/SimpleGPUHashTable
template<class Key, class Value>
class LinearProbingHashTable {
public:
    static constexpr float maxLoadFactor = 0.7;
    static constexpr Key emptyMarker = std::numeric_limits<Key>::max();

    struct Entry {
        Key key;
        Value value;
    };

    struct DeviceHandle {
        Entry* table = nullptr;
        const device_size_t capacity;

        __device__ void insert(Key key, Value value) {
            device_size_t slot = murmur3_hash(key) & (capacity - 1);
            while (true) {
                //device_size_t prev = atomicCAS(&table[slot].key, emptyMarker, key);
                device_size_t prev = tmpl_atomic_cas(&table[slot].key, emptyMarker, key);
                if (prev == emptyMarker || prev == key) {
                    table[slot].value = value;
                    return;
                }
                slot = (slot + 1) & (capacity - 1);
            }
        }

        __device__ bool lookup(Key key, Value& value) {
            device_size_t slot = murmur3_hash(key) & (capacity - 1);
            while (true) {
                Entry* entry = &table[slot];
                if (entry->key == key) {
                    value = entry->value;
                    return true;
                } else if (entry->key == emptyMarker) {
                    return false;
                }
                slot = (slot + 1) & (capacity - 1);
            }
        }
    } deviceHandle;

    static constexpr device_size_t calculateTableSize(device_size_t occupancyUpperBound) {
        float sizeHint = static_cast<float>(occupancyUpperBound) / maxLoadFactor;
        device_size_t n = static_cast<device_size_t>(std::ceil(std::log2(sizeHint))); // find n such that: 2^n >= sizeHint
        device_size_t tableSize = 1ul << n;
        return tableSize;
    }

    LinearProbingHashTable(device_size_t occupancyUpperBound)
        : deviceHandle{ nullptr, calculateTableSize(occupancyUpperBound) }
    {
        auto ret = cudaMalloc(&deviceHandle.table, deviceHandle.capacity*sizeof(Entry));
        assert(ret == cudaSuccess);
        ret = cudaMemset(deviceHandle.table, 0xff, deviceHandle.capacity*sizeof(Entry));
        assert(ret == cudaSuccess);
    }

    ~LinearProbingHashTable() {
        cudaFree(deviceHandle.table);
    }
};
