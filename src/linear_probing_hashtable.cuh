#pragma once

#include <cstring>
#include <limits>
#include <memory>

#include <fast-interconnects/gpu_common.h>

#include "cuda_utils.cuh"
#include "device_definitions.hpp"

// simple linear probing hashtable
// implementation is based on: https://github.com/nosferalatu/SimpleGPUHashTable
template<class Key, class Value>
class linear_probing_hashtable {
public:
    static constexpr float max_load_factor = 0.5;
    static constexpr Key empty_marker = std::numeric_limits<Key>::max();

    struct entry {
        Key key;
        Value value;
    };

    struct mutable_data {
        uint64_t counter = 0u;
    };

    struct device_handle {
        entry* const table = nullptr;
        const device_size_t capacity = 0u;

        mutable_data* const mutable_data_ptr = nullptr;
    } _device_handle_inst;

    /**
     * @brief Inserts the provided key-value pair into the hash table. In case
     * there already exits an entry with an identical key, the value is
     * replaced with the provided value.
     * 
     * @param handle_inst 
     * @param key 
     * @param value 
     */
    __device__ static void insert(const device_handle& handle_inst, Key key, Value value) {
        device_size_t slot = murmur3_hash(key) & (handle_inst.capacity - 1u);
        for (device_size_t i = 0; i < handle_inst.capacity; ++i) {
            device_size_t prev = tmpl_atomic_cas(&handle_inst.table[slot].key, empty_marker, key);
            if (prev == empty_marker || prev == key) {
                handle_inst.table[slot].value = value;
                return;
            }
            slot = (slot + 1u) & (handle_inst.capacity - 1u);
        }
    }

    __device__ static bool lookup(const device_handle& handle_inst, Key key, Value& value) {
        device_size_t slot = murmur3_hash(key) & (handle_inst.capacity - 1u);
        while (true) {
            /*
            entry* entry = &table[slot];
            if (entry->key == key) {
                value = entry->value;
                return true;
            } else if (entry->key == empty_marker) {
                return false;
            }
            slot = (slot + 1) & (capacity - 1);*/
            longlong2 tmp = ptx_load_cache_streaming(reinterpret_cast<longlong2 const *>(&handle_inst.table[slot]));
            const Key slot_key = tmp.x;
            if (slot_key == key) {
                value = tmp.y;
                return true;
            } else if (slot_key == empty_marker) {
                return false;
            }
            slot = (slot + 1u) & (handle_inst.capacity - 1u);
        }
    }

    static constexpr device_size_t calculate_table_size(device_size_t occupancy_upper_bound) {
        float size_hint = static_cast<float>(occupancy_upper_bound) / max_load_factor;
        device_size_t n = static_cast<device_size_t>(std::ceil(std::log2(size_hint))); // find n such that: 2^n >= size_hint
        device_size_t table_size = 1u << n;
        return table_size;
    }

    linear_probing_hashtable(device_size_t occupancy_upper_bound) {
        const auto capacity = calculate_table_size(occupancy_upper_bound);
        entry* table;
        auto ret = cudaMalloc(&table, capacity*sizeof(entry));
        assert(ret == cudaSuccess);
        ret = cudaMemset(table, 0xff, capacity*sizeof(entry));
        assert(ret == cudaSuccess);

        device_handle actual_handle { table, capacity, nullptr };
        std::memcpy(&_device_handle_inst, &actual_handle, sizeof(actual_handle));
    }

    ~linear_probing_hashtable() {
        cudaFree(_device_handle_inst.table);
    }
};
