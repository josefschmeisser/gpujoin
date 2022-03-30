#pragma once

#include <sys/types.h>
#include <cstdint>
#include <numeric>
#include <type_traits>

#include "search.cuh"
#include "btree.cuh"
#include "harmonia.cuh"
#include "rs.cuh"
#include "vector_view.hpp"

/*
Each index structure struct contains a device_index struct which upon kernel invocation will be passed by value.
Passing a struct by value has the advantage that the CUDA runtime copies the entire struct into constant memory.
Reads to members of these structs are therefore cached and can futhermore be broadcasted/mutlicasted when accessed by multiple threads.
*/

enum class index_type_enum : unsigned { btree, harmonia, lower_bound, radixspline, no_op };

/*
template<class >
struct 
*/

template<class Key>
struct abstract_index {
    using key_t = Key;

    __host__ virtual void construct(const vector_view<key_t>& h_column, const key_t* d_column) = 0;

    __host__ virtual size_t memory_consumption() const = 0;
/*
    template<index_type_enum index_type>
    __host__ typename as();

    __host__*/
};


template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct btree_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    using btree_t = index_structures::btree<key_t, value_t, HostAllocator, invalid_tid>;

    btree_t h_tree_;
    device_array_wrapper<typename btree_t::page> h_guard;

    struct device_index_t {
        const typename btree_t::NodeBase* d_tree_;

        __device__ __forceinline__ value_t lookup(const key_t key) const {
            return btree_t::lookup(d_tree_, key);
            //return btree_t::lookup_with_hints(tree_, key); // TODO
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            return btree_t::cooperative_lookup(active, d_tree_, key);
        }
    } device_index;

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        h_tree_.construct(h_column, 0.7);

        if /*constexpr*/ (std::is_same<DeviceAllocator<int>, HostAllocator<int>>::value) {
            printf("no migration necessary\n");
            device_index.d_tree_ = h_tree_.root;
        } else {
            printf("migrating btree...\n");
            DeviceAllocator<key_t> device_allocator;
            device_index.d_tree_ = h_tree_.migrate(device_allocator, h_guard);
        }
    }

    __host__ size_t memory_consumption() const override {
        return h_tree_.tree_size_in_byte(h_tree_.root);
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct type_name<btree_index<Key, Value, DeviceAllocator, HostAllocator>> {
    static const char* value() {
        return "btree";
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct radix_spline_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;

    size_t memory_consumption_ = 0;
    rs::device_array_guard<key_t> guard_;
    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_index_t {
        const key_t* d_column_;
        rs::DeviceRadixSpline<key_t> d_rs_;

        __device__ __forceinline__ value_t lookup(const key_t key) const {
            const unsigned estimate = rs::cuda::get_estimate(&d_rs_, key); // FIXME accessing this member by a pointer will result in unached global loads
            const unsigned begin = (estimate < d_rs_.max_error_) ? 0 : (estimate - d_rs_.max_error_);
            const unsigned end = (estimate + d_rs_.max_error_ + 2 > d_rs_.num_keys_) ? d_rs_.num_keys_ : (estimate + d_rs_.max_error_ + 2);

            const auto bound_size = end - begin;
            const unsigned pos = begin + rs::cuda::lower_bound(key, &d_column_[begin], bound_size, [] (const key_t& a, const key_t& b) -> int {
                return a < b;
            });
            return (pos < d_rs_.num_keys_) ? static_cast<value_t>(pos) : invalid_tid;
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            if (active) {
                return lookup(key);
            } else {
                return value_t();
            }
/*
            assert(false); // TODO implement
            return value_t();*/
        }
    } device_index;

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        device_index.d_column_ = d_column;
        auto h_rs = rs::build_radix_spline(h_column);

        // migrate radix spline
        const auto start = std::chrono::high_resolution_clock::now();

        DeviceAllocator<key_t> device_allocator;
        guard_ = migrate_radix_spline(h_rs, device_index.d_rs_, device_allocator);

        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()/1000.;
        std::cout << "radixspline migration time: " << duration << " ms\n";

        // calculate memory consumption
        memory_consumption_ = sizeof(rs::DeviceRadixSpline<key_t>) +
            guard_.radix_table_guard.size()*sizeof(typename decltype(guard_.radix_table_guard)::value_type) +
            guard_.spline_points_guard.size()*sizeof(typename decltype(guard_.spline_points_guard)::value_type);

        auto rrs __attribute__((unused)) = reinterpret_cast<const rs::RawRadixSpline<key_t>*>(&h_rs);
        assert(h_column.size() == rrs->num_keys_);
    }

    __host__ size_t memory_consumption() const override {
        return memory_consumption_;
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct type_name<radix_spline_index<Key, Value, DeviceAllocator, HostAllocator>> {
    static const char* value() {
        return "radix_spline";
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct harmonia_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    using harmonia_t = harmonia::harmonia_tree<key_t, value_t, HostAllocator, 32 + 1, invalid_tid>;

    harmonia_t tree;
    typename harmonia_t::memory_guard_t guard;

    struct device_index_t {
        typename harmonia_t::device_handle_t d_tree;

        __device__ __forceinline__ value_t lookup(const key_t key) const {
            assert(false); // not available
            return value_t();
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            //return harmonia_t::lookup<4>(active, d_tree, key);
            //return harmonia_t::ntg_lookup(active, d_tree, key);
            return harmonia_t::ntg_lookup_with_caching(active, d_tree, key);
        }
    } device_index;

    __host__ virtual void construct(const vector_view<key_t>& h_column, const key_t* /*d_column*/) override {
        tree.construct(h_column);
        DeviceAllocator<key_t> device_allocator;
        tree.create_device_handle(device_index.d_tree, device_allocator, guard);
    }

    __host__ size_t memory_consumption() const override {
        auto size = guard.keys_guard.size()*sizeof(typename decltype(guard.keys_guard)::value_type);
        size += guard.children_guard.size()*sizeof(typename decltype(guard.children_guard)::value_type);
        if (guard.values_guard.data()) {
            size += guard.values_guard.size()*sizeof(typename decltype(guard.values_guard)::value_type);
        }
        return size;
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct type_name<harmonia_index<Key, Value, DeviceAllocator, HostAllocator>> {
    static const char* value() {
        return "harmonia";
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct lower_bound_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_index_t {
        const key_t* d_column;
        unsigned d_size;

        __device__ __forceinline__ value_t lookup(const key_t key) const {
            //auto pos = branchy_binary_search(key, d_column, d_size);
            auto pos = branch_free_binary_search(key, d_column, d_size);
            return (pos < d_size) ? static_cast<value_t>(pos) : invalid_tid;
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            // TODO implement cooperative binary search

            if (active) {
                //auto pos = branchy_binary_search(key, d_column, d_size);
                auto pos = branch_free_binary_search(key, d_column, d_size);
                return (pos < d_size) ? static_cast<value_t>(pos) : invalid_tid;
            } else {
                return value_t();
            }
        }
    } device_index;

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        device_index.d_column = d_column;
        device_index.d_size = h_column.size();
    }

    __host__ size_t memory_consumption() const override {
        return 0;
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct type_name<lower_bound_index<Key, Value, DeviceAllocator, HostAllocator>> {
    static const char* value() {
        return "lower_bound";
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct no_op_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_index_t {
        const key_t* d_column;
        unsigned d_size;

        __device__ __forceinline__ value_t lookup(const key_t key) const { return value_t(); }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const { return value_t(); }
    } device_index;

    __host__ void construct(const vector_view<key_t>& /*h_column*/, const key_t* /*d_column*/) override {}

    __host__ size_t memory_consumption() const override {
        return 0;
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct type_name<no_op_index<Key, Value, DeviceAllocator, HostAllocator>> {
    static const char* value() {
        return "no_op";
    }
};

// traits
template<class IndexType>
struct requires_sorted_input;
