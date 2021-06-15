#pragma once

#include <sys/types.h>
#include "search.cuh"

/* TODO
#include "btree.cuh"
#include "btree.cu"
*/
#include "harmonia.cuh"
#include "rs.cuh"

#if 0
struct btree_index {
    const btree::Node* tree_;

    __host__ void construct(const std::vector<btree::key_t>& h_column, const btree::key_t* d_column) {
        auto tree = btree::construct(h_column, 0.7);
        if (prefetch_index) {
            btree::prefetchTree(tree, 0);
        }
        tree_ = tree;
    }

    __device__ __forceinline__ btree::payload_t operator() (const btree::key_t key) const {
        return btree::cuda::btree_lookup(tree_, key);
    //    return btree::cuda::btree_lookup_with_hints(tree_, key); // TODO
    }

    __device__ __forceinline__ btree::payload_t cooperative_lookup(bool active, const btree::key_t key) const {
        return btree::cuda::btree_cooperative_lookup(active, tree_, key);
    }
};
#endif

template<class Key, class Value>
struct radix_spline_index {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    rs::DeviceRadixSpline<key_t>* d_rs_;
    const key_t* d_column_;

    __host__ void construct(const std::vector<key_t>& h_column, const key_t* d_column) {
        d_column_ = d_column;
        auto h_rs = rs::build_radix_spline(h_column);

        // copy radix spline
        const auto start = std::chrono::high_resolution_clock::now();
        d_rs_ = rs::copy_radix_spline<vector_to_device_array>(h_rs); // TODO
        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()/1000.;
        std::cout << "radixspline transfer time: " << duration << " ms\n";

        auto rrs __attribute__((unused)) = reinterpret_cast<const rs::RawRadixSpline<key_t>*>(&h_rs);
        assert(h_column.size() == rrs->num_keys_);
    }

    __device__ __forceinline__ value_t operator() (const key_t key) const {
        const unsigned estimate = rs::cuda::get_estimate(d_rs_, key);
        const unsigned begin = (estimate < d_rs_->max_error_) ? 0 : (estimate - d_rs_->max_error_);
        const unsigned end = (estimate + d_rs_->max_error_ + 2 > d_rs_->num_keys_) ? d_rs_->num_keys_ : (estimate + d_rs_->max_error_ + 2);

        const auto bound_size = end - begin;
        const unsigned pos = begin + rs::cuda::lower_bound(key, &d_column_[begin], bound_size, [] (const key_t& a, const key_t& b) -> int {
            return a < b;
        });
        return (pos < d_rs_->num_keys_) ? static_cast<value_t>(pos) : invalid_tid;
    }
};

template<class Key, class Value>
struct harmonia_index {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    using harmonia_t = harmonia::harmonia_tree<key_t, value_t, 32 + 1, invalid_tid>;

    harmonia_t tree;
    const typename harmonia_t::device_handle_t* __restrict__ d_tree;

    __host__ void construct(const std::vector<key_t>& h_column, const key_t* /*d_column*/) {
        tree.construct(h_column);
        tree.create_device_handle();
        d_tree = tree.device_handle;
    }

//    __device__ __forceinline__ value_t lookup(const key_t key) const;

    __device__ __forceinline__ value_t cooperative_lookup(bool active, key_t key) const {
        return harmonia_t::lookup<4>(active, d_tree, key);
    }
};

template<class Key, class Value>
struct lower_bound_index {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_data_t {
        const key_t* d_column;
        const unsigned d_size;
    }* device_data;

    __host__ void construct(const std::vector<key_t>& h_column, const key_t* d_column) {
        device_data_t tmp { d_column, static_cast<unsigned>(h_column.size()) };
        cudaMalloc(&device_data, sizeof(device_data_t));
        cudaMemcpy(device_data, &tmp, sizeof(device_data_t), cudaMemcpyHostToDevice);
    }

    __device__ __forceinline__ value_t lookup(const key_t key) const {
//        auto pos = branchy_binary_search(key, device_data->d_column, device_data->d_size);
        auto pos = branch_free_binary_search(key, device_data->d_column, device_data->d_size);
        return (pos < device_data->d_size) ? static_cast<value_t>(pos) : invalid_tid;
    }
};

// traits
template<class IndexType>
struct requires_sorted_input;
