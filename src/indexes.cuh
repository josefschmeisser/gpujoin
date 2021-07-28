#pragma once

#include <sys/types.h>
#include <cstdint>
#include <numeric>

#include "search.cuh"
#include "btree.cuh"
#include "harmonia.cuh"
#include "rs.cuh"

/*
Each index structure struct contains a device_index struct which upon kernel invocation will be passed by value.
Passing a struct by value has the advantage that the CUDA runtime copies the entire struct into constant memory.
Reads to members of these structs are therefore cached and can futhermore be broadcasted/mutlicasted when accessed by multiple threads.
*/

#if 1
template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct btree_index {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    using btree_t = index_structures::btree<key_t, value_t, invalid_tid>;

    btree_t h_tree_;

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

    template<class Vector>
    __host__ void construct(const Vector& h_column, const key_t* d_column) {
        h_tree_.construct(h_column, 0.7); // TODO use HostAllocator
        device_index.d_tree_ = h_tree_.copy_btree_to_gpu(h_tree_.root); // TODO use DeviceAllocator
/*
        if (prefetch_index) {
            btree::prefetchTree(tree, 0);
        }
        tree_ = tree;*/
    }

    template<class Allocator>
    __host__ void construct(const std::vector<key_t>& h_column, const key_t* d_column, Allocator& allocator);
    // TODO
};
#endif

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct radix_spline_index {
    using key_t = Key;
    using value_t = Value;

    rs::device_array_guard<key_t> guard;
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
            assert(false); // TODO implement
            return value_t();
        }
    } device_index;

    template<class Vector>
    __host__ void construct(const Vector& h_column, const key_t* d_column) {
        device_index.d_column_ = d_column;
        auto h_rs = rs::build_radix_spline(h_column);

        // migrate radix spline
        const auto start = std::chrono::high_resolution_clock::now();

//        device_index.d_rs_ = rs::copy_radix_spline<vector_to_device_array>(h_rs); // TODO
        DeviceAllocator<key_t> device_allocator;
        migrate_radix_spline(h_rs, device_index.d_rs_, device_allocator);

        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()/1000.;
        std::cout << "radixspline migration time: " << duration << " ms\n";

        auto rrs __attribute__((unused)) = reinterpret_cast<const rs::RawRadixSpline<key_t>*>(&h_rs);
        assert(h_column.size() == rrs->num_keys_);
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct harmonia_index {
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

    template<class Vector>
    __host__ void construct(const Vector& h_column, const key_t* /*d_column*/) {
        tree.construct(h_column);
        DeviceAllocator<key_t> device_allocator;
        tree.create_device_handle(device_index.d_tree, device_allocator, guard);
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct lower_bound_index {
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
            //auto pos = branchy_binary_search(key, d_column, d_size);
            auto pos = branch_free_binary_search(key, d_column, d_size);
            return (pos < d_size) ? static_cast<value_t>(pos) : invalid_tid;
        }
    } device_index;

    template<class Vector>
    __host__ void construct(const Vector& h_column, const key_t* d_column) {
        device_index.d_column = d_column;
        device_index.d_size = h_column.size();
    }
};

// traits
template<class IndexType>
struct requires_sorted_input;
