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
                    table[slot].value = value;
                    return;
                }
                slot = (slot + 1) & (capacity - 1);
            }
        }

        __device__ bool lookup(Key key, Value& value) {
            uint32_t slot = hash(key);
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

    static constexpr uint32_t calculateTableSize(uint32_t occupancyUpperBound) {
        float sizeHint = static_cast<float>(occupancyUpperBound) / maxLoadFactor;
        uint32_t n = static_cast<uint32_t>(std::ceil(std::log2(sizeHint))); // find n such that: 2^n >= sizeHint
        uint32_t tableSize = 1ul << n;
        return tableSize;
    }

    LinearProbingHashTable(uint32_t occupancyUpperBound)
        : deviceHandle{nullptr, calculateTableSize(occupancyUpperBound)}
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
