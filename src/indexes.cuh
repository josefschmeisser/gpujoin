#pragma once

#include "indexes.hpp"

#include <sys/types.h>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "search.cuh"
#include "btree.cuh"
#include "harmonia.cuh"
#include "rs.cuh"
#include "vector_view.hpp"
#include "device_definitions.hpp"

/*
Each index structure struct contains a device_index struct which upon kernel invocation will be passed by value.
Passing a struct by value has the advantage that the CUDA runtime copies the entire struct into constant memory.
Reads to members of these structs are therefore cached and can futhermore be broadcasted/mutlicasted when accessed by multiple threads.
*/

/*
enum class index_type_enum : unsigned { btree, harmonia, binary_search, radix_spline, no_op };

index_type_enum parse_index_type(const std::string& index_name);
*/

template<class Key>
struct abstract_index {
    using key_t = Key;

    __host__ virtual void construct(const vector_view<key_t>& h_column, const key_t* d_column) = 0;

    __host__ virtual size_t memory_consumption() const = 0;
};

template<class BtreeType>
struct btree_info {
    using btree_type = BtreeType;
    using key_type = typename BtreeType::key_t;
    using value_type = typename BtreeType::value_t;
    using node_base_type = typename BtreeType::NodeBase;
};

struct btree_lookup_algorithm {
    //static constexpr char name[] = "lookup";
    static constexpr const char* name() {
        return "lookup";
    }

    template<class BtreeInfoType>
    __device__ __forceinline__ auto operator() (const BtreeInfoType btree_info, const typename BtreeInfoType::node_base_type* tree, typename BtreeInfoType::key_type key) const {
        return BtreeInfoType::btree_type::lookup(tree, key);
    }
};

struct btree_cooperative_lookup_algorithm {
    static constexpr const char* name() {
        return "cooperative_lookup";
    }

    template<class BtreeInfoType>
    __device__ __forceinline__ auto operator() (const BtreeInfoType btree_info, bool active, const typename BtreeInfoType::node_base_type* tree, typename BtreeInfoType::key_type key) const {
        return BtreeInfoType::btree_type::cooperative_lookup(active, tree, key);
    }
};

template<class LookupAlgorithm>
struct btree_pseudo_cooperative_lookup_algorithm {
    static constexpr const char* name() {
        //return "pseudo_cooperative_lookup";
        return LookupAlgorithm::name();
    }

    template<class BtreeInfoType>
    __device__ __forceinline__ auto operator() (const BtreeInfoType btree_info, bool active, const typename BtreeInfoType::node_base_type* tree, typename BtreeInfoType::key_type key) const {
        //return BtreeInfoType::btree_type::lookup(tree, key);
        static const LookupAlgorithm lookup_algorithm{};
        return lookup_algorithm(btree_info, tree, key);
    }
};

struct default_btree_index_configuration {
    using lookup_algorithm_type = btree_lookup_algorithm;
    using cooperative_lookup_algorithm_type = btree_pseudo_cooperative_lookup_algorithm<lookup_algorithm_type>;
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator, class IndexConfiguration = default_btree_index_configuration>
struct btree_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;
    using index_configuration_t = IndexConfiguration;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    using btree_t = index_structures::btree<key_t, value_t, HostAllocator, invalid_tid>;
    using btree_info_type = btree_info<btree_t>;

    btree_t h_tree_;
    device_array_wrapper<typename btree_t::page> h_guard;

    struct device_index_t {
        const typename btree_t::NodeBase* d_tree_;

        static constexpr btree_info_type btree_info_inst{};

        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            static const typename IndexConfiguration::lookup_algorithm_type lookup_algorithm{};
            return lookup_algorithm(btree_info_inst, d_tree_, key);
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            static const typename IndexConfiguration::cooperative_lookup_algorithm_type lookup_algorithm{};
            return lookup_algorithm(btree_info_inst, active, d_tree_, key);
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

template<class SearchAlgorithm>
struct pseudo_cooperative_search_algorithm {
    static constexpr const char* name() {
        return SearchAlgorithm::name();
    }

    template<class T1, class T2, class Compare = device_less<T1>>
    __device__ __forceinline__ device_size_t operator() (bool active, const T1& x, const T2* arr, const device_size_t size, Compare cmp = device_less<T1>{}) const {
        static const SearchAlgorithm search_algorithm{};
        return search_algorithm(x, arr, size);
    }
};

struct default_radix_spline_index_configuration {
    using search_algorithm_type = branchy_lower_bound_search_algorithm;
    using cooperative_search_algorithm_type = pseudo_cooperative_search_algorithm<search_algorithm_type>;
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator, class IndexConfiguration = default_radix_spline_index_configuration>
struct radix_spline_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;
    using index_configuration_t = IndexConfiguration;

    static_assert(sizeof(value_t) <= sizeof(device_size_t));

    size_t memory_consumption_ = 0;
    rs::device_array_guard<key_t> guard_;
    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_index_t {
        const key_t* d_column_;
        rs::DeviceRadixSpline<key_t> d_rs_;

        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            static const typename IndexConfiguration::search_algorithm_type search_algorithm{};

            const double estimate = rs::cuda::get_estimate(d_rs_, key); // FIXME accessing this member by a pointer will result in uncached global loads
            const device_size_t begin = (estimate < d_rs_.max_error_) ? 0 : (estimate - d_rs_.max_error_);
            const device_size_t end = (estimate + d_rs_.max_error_ + 2 > d_rs_.num_keys_) ? d_rs_.num_keys_ : (estimate + d_rs_.max_error_ + 2);

            const device_size_t bound_size = end - begin;
            const device_size_t pos = begin + search_algorithm(key, &d_column_[begin], bound_size);
            return (pos < d_rs_.num_keys_) ? static_cast<value_t>(pos) : invalid_tid;
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            static const typename IndexConfiguration::cooperative_search_algorithm_type search_algorithm{};

            const double estimate = rs::cuda::get_estimate(d_rs_, key); // FIXME accessing this member by a pointer will result in uncached global loads
            const device_size_t begin = (estimate < d_rs_.max_error_) ? 0 : (estimate - d_rs_.max_error_);
            const device_size_t end = (estimate + d_rs_.max_error_ + 2 > d_rs_.num_keys_) ? d_rs_.num_keys_ : (estimate + d_rs_.max_error_ + 2);

            const device_size_t bound_size = end - begin;
            const device_size_t pos = begin + search_algorithm(active, key, &d_column_[begin], bound_size);
            return (active && pos < d_rs_.num_keys_) ? static_cast<value_t>(pos) : invalid_tid;
        }
    } device_index;

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        if (h_column.size() >= invalid_tid) {
            throw std::runtime_error("'value_t' does not suffice");
        }

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

        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            assert(false); // not available
            return invalid_tid;
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            //return harmonia_t::lookup<4>(active, d_tree, key);
            //return harmonia_t::ntg_lookup(active, d_tree, key);
            return harmonia_t::ntg_lookup_with_caching(active, d_tree, key);
        }
    } device_index;

    __host__ virtual void construct(const vector_view<key_t>& h_column, const key_t* /*d_column*/) override {
        tree.construct(h_column);

        // migrate harmonia spline
        const auto start = std::chrono::high_resolution_clock::now();

        DeviceAllocator<key_t> device_allocator;
        tree.create_device_handle(device_index.d_tree, device_allocator, guard);

        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()/1000.;
        std::cout << "harmonia migration time: " << duration << " ms\n";
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

template<class SearchAlgorithm>
struct binary_search_index_pseudo_cooperative_search_algorithm {
    static constexpr const char* name() {
        return SearchAlgorithm::name();
    }

    template<class T>
    __device__ __forceinline__ device_size_t operator() (bool active, T x, const T* arr, const device_size_t size) const {
        static const SearchAlgorithm search_algorithm{};
        return search_algorithm(x, arr, size);
    }
};

struct default_binary_search_index_configuration {
    //using search_algorithm_type = branch_free_binary_search_algorithm;
    using search_algorithm_type = branchy_lower_bound_search_algorithm;
    //using cooperative_search_algorithm_type = binary_search_index_pseudo_cooperative_search_algorithm<search_algorithm_type>;
    using cooperative_search_algorithm_type = cooperative_binary_search_algorithm;
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator, class IndexConfiguration = default_binary_search_index_configuration>
struct binary_search_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;
    using index_configuration_t = IndexConfiguration;

    static_assert(sizeof(value_t) <= sizeof(device_size_t));

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_index_t {
        const key_t* d_column;
        device_size_t d_size;

        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            static const typename IndexConfiguration::search_algorithm_type search_algorithm{};
            const auto pos = search_algorithm(key, d_column, d_size);
            return (pos < d_size) ? static_cast<value_t>(pos) : invalid_tid;
        }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const {
            static const typename IndexConfiguration::cooperative_search_algorithm_type cooperative_search_algorithm{};
            const auto pos = cooperative_search_algorithm(active, key, d_column, d_size);
            return (active && pos < d_size) ? static_cast<value_t>(pos) : invalid_tid;
        }
    } device_index;

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        device_index.d_column = d_column;
        device_index.d_size = h_column.size();
        if (device_index.d_size >= invalid_tid) {
            throw std::runtime_error("'value_t' does not suffice");
        }
    }

    __host__ size_t memory_consumption() const override {
        return 0;
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator, class IndexConfiguration>
struct type_name<binary_search_index<Key, Value, DeviceAllocator, HostAllocator, IndexConfiguration>> {
    static const char* value() {
        return "binary_search";
    }
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator>
struct no_op_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    struct device_index_t {
        const key_t* d_column;
        device_size_t d_size;

        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const { return invalid_tid; }

        __device__ __forceinline__ value_t cooperative_lookup(const bool active, const key_t key) const { return invalid_tid; }
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
