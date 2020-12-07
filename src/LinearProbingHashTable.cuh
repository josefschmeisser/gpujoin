#pragma once

#include <limits>

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
        const uint32_t capacity;

        __device__ uint32_t hash(Key k) {
            static_assert(sizeof(Key) == 4);
            k ^= k >> 16;
            k *= 0x85ebca6b;
            k ^= k >> 13;
            k *= 0xc2b2ae35;
            k ^= k >> 16;
            return k & (capacity - 1);
        }

        __device__ void insert(Key key, Value value) {
            uint32_t slot = hash(key);
            while (true) {
                uint32_t prev = atomicCAS(&table[slot].key, emptyMarker, key);
                if (prev == emptyMarker || prev == key) {
                    table[slot]->value = value;
                    return;
                }
                slot = (slot + 1) & (capacity - 1);
            }
        }

        __device__ bool lookup(Key key, Value& value) {
            uint32_t slot = hash(key);
            while (true) {
                Entry* entry = table[slot];
                if (entry->key == key) {
                    entry->value;
                    return true;
                } else if (entry->key == emptyMarker) {
                    return false;
                }
                slot = (slot + 1) & (capacity - 1);
            }
        }
    } deviceHandle;

    static constexpr size_t calculateTableSize(size_t occupancyUpperBound) {
        float sizeHint = static_cast<float>(occupancyUpperBound) / maxLoadFactor;
        size_t n = static_cast<size_t>(std::ceil(std::log2(sizeHint))); // find n such that: 2^n >= sizeHint
        size_t tableSize = 1ul << n;
        return tableSize;
    }

    LinearProbingHashTable(size_t occupancyUpperBound) 
        : deviceHandle{nullptr, calculateTableSize(occupancyUpperBound)}
    {
        auto ret = cudaMalloc(&deviceHandle.table, deviceHandle.capacity*sizeof(DeviceHandle::Entry));
        assert(ret == cudaSuccess);
    }

    ~LinearProbingHashTable() {
        cudaFree(deviceHandle.table);
    }
};
