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

    virtual ~abstract_index() = default;

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

    template<bool IsConst = false>
    struct device_handle_t {
        //const typename btree_t::NodeBase* d_tree_;
        add_const_if_t<typename btree_t::NodeBase*, IsConst> d_tree_;

        static constexpr btree_info_type btree_info_inst{};

        /*
        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            static const typename IndexConfiguration::lookup_algorithm_type lookup_algorithm{};
            return lookup_algorithm(btree_info_inst, d_tree_, key);
        }
        */
    };
    device_handle_t<false> _device_handle_inst;

    ~btree_index() override = default;

    __device__ __forceinline__ static value_t device_cooperative_lookup(const device_handle_t<true>& handle_inst, const bool active, const key_t key) {
        static const typename IndexConfiguration::cooperative_lookup_algorithm_type lookup_algorithm{};
        return lookup_algorithm(handle_inst.btree_info_inst, active, handle_inst.d_tree_, key);
    }

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        h_tree_.construct(h_column, 0.7);

        if /*constexpr*/ (std::is_same<DeviceAllocator<int>, HostAllocator<int>>::value) {
            printf("no migration necessary\n");
            _device_handle_inst.d_tree_ = h_tree_.root;
        } else {
            printf("migrating btree...\n");
            DeviceAllocator<key_t> device_allocator;
            _device_handle_inst.d_tree_ = h_tree_.migrate(device_allocator, h_guard);
        }
    }

    __host__ size_t memory_consumption() const override {
        return h_tree_.tree_size_in_byte(h_tree_.root);
    }

    __host__ device_handle_t<true>& get_device_handle() {
        //return _device_handle_inst;
        auto handle = reinterpret_cast<device_handle_t<true>*>(&_device_handle_inst);
        return *handle;
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
    //using search_algorithm_type = branch_free_lower_bound_search_algorithm;
    using cooperative_search_algorithm_type = pseudo_cooperative_search_algorithm<search_algorithm_type>;
    //using cooperative_search_algorithm_type = cooperative_binary_search_algorithm;
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

    template<bool IsConst = false>
    struct device_handle_t {
        //typename add_const_if<key_t* const, IsConst>::type d_column_ = nullptr;
        //key_t* const d_column_ = nullptr;
        //const key_t* const d_column_ = nullptr;
        add_const_if_t<const key_t* __restrict__, IsConst> d_column_;

        //const rs::DeviceRadixSpline<key_t> d_rs_;
        //rs::DeviceRadixSpline<key_t> d_rs_;
        //add_const_if_t<rs::DeviceRadixSpline<key_t>, IsConst> d_rs_;
        add_const_if_t<typename rs::device_radix_spline_tmpl<key_t, IsConst>, IsConst> d_rs_;
/*
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
*/
    //} _device_handle_inst;
    };
    device_handle_t<false> _device_handle_inst;
    /*
    struct device_handle {
        entry* const table = nullptr;
        const device_size_t capacity = 0u;

        mutable_data* const mutable_data_ptr = nullptr;
    } _device_handle_inst;

    __device__ static void insert(const device_handle& handle_inst, Key key, Value value) {
    */

    ~radix_spline_index() override = default;

    __device__ __forceinline__ static value_t device_cooperative_lookup(const device_handle_t<true>& handle_inst, const bool active, const key_t key) {
        static const typename IndexConfiguration::cooperative_search_algorithm_type search_algorithm{};

        auto bounds = rs::cuda::find_bounds(handle_inst.d_rs_, key);
        const device_size_t begin = bounds.begin;
        const device_size_t end = bounds.end;
        const device_size_t bound_size = end - begin;
        const device_size_t pos = begin + search_algorithm(active, key, &handle_inst.d_column_[begin], bound_size);
        return (active && pos < handle_inst.d_rs_.num_keys_) ? static_cast<value_t>(pos) : invalid_tid;
    }

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        if (h_column.size() >= invalid_tid) {
            throw std::runtime_error("'value_t' does not suffice");
        }

        _device_handle_inst.d_column_ = d_column;
        auto h_rs = rs::build_radix_spline(h_column);

        // migrate radix spline
        const auto start = std::chrono::high_resolution_clock::now();
        DeviceAllocator<key_t> device_allocator;
        guard_ = migrate_radix_spline(h_rs, _device_handle_inst.d_rs_, device_allocator);
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

    __host__ device_handle_t<true>& get_device_handle() {
        //return _device_handle_inst;
        //return device_handle_t<true> {};
        auto handle = reinterpret_cast<device_handle_t<true>*>(&_device_handle_inst);
        return *handle;
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

    template<bool IsConst = false>
    struct device_handle_t {
        //typename harmonia_t::device_handle_t d_tree;
        add_const_if_t<typename harmonia_t::const_device_handle_t, IsConst> d_tree;

        /*
        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            assert(false); // not available
            return invalid_tid;
        }
        */
    };
    device_handle_t<false> _device_handle_inst;

    ~harmonia_index() override = default;

    __device__ __forceinline__ static value_t device_cooperative_lookup(const device_handle_t<true>& handle_inst, const bool active, const key_t key) {
        //return harmonia_t::lookup<4>(active, d_tree, key);
        //return harmonia_t::ntg_lookup(active, d_tree, key);
        return harmonia_t::ntg_lookup_with_caching(active, handle_inst.d_tree, key);
    }

    __host__ virtual void construct(const vector_view<key_t>& h_column, const key_t* /*d_column*/) override {
        tree.construct(h_column);

        // migrate harmonia
        const auto start = std::chrono::high_resolution_clock::now();

        DeviceAllocator<key_t> device_allocator;
        //tree.create_device_handle(_device_handle_inst.d_tree, device_allocator, guard);
        auto tmp_handle = tree.create_device_handle(device_allocator, guard);
        std::memcpy(&_device_handle_inst.d_tree, &tmp_handle, sizeof(_device_handle_inst.d_tree));

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

    __host__ device_handle_t<true>& get_device_handle() {
        //return _device_handle_inst;
        auto handle = reinterpret_cast<device_handle_t<true>*>(&_device_handle_inst);
        return *handle;
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
    using search_algorithm_type = branchy_lower_bound_search_algorithm;
    //using search_algorithm_type = branch_free_lower_bound_search_algorithm;
    using cooperative_search_algorithm_type = binary_search_index_pseudo_cooperative_search_algorithm<search_algorithm_type>;
    //using cooperative_search_algorithm_type = cooperative_binary_search_algorithm;
};

template<class Key, class Value, template<class T> class DeviceAllocator, template<class T> class HostAllocator, class IndexConfiguration = default_binary_search_index_configuration>
struct binary_search_index : public abstract_index<Key> {
    using key_t = Key;
    using value_t = Value;
    using index_configuration_t = IndexConfiguration;

    static_assert(sizeof(value_t) <= sizeof(device_size_t));

    static const value_t invalid_tid = std::numeric_limits<value_t>::max();

    template<bool IsConst = false>
    struct device_handle_t {
        //const key_t* d_column;
        add_const_if_t<const key_t* __restrict__, IsConst> d_column;
        //device_size_t d_size;
        add_const_if_t<device_size_t, IsConst> d_size;

        /*
        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const {
            static const typename IndexConfiguration::search_algorithm_type search_algorithm{};
            const auto pos = search_algorithm(key, d_column, d_size);
            return (pos < d_size) ? static_cast<value_t>(pos) : invalid_tid;
        }
        */
    };
    device_handle_t<false> _device_handle_inst;

    ~binary_search_index() override = default;

    __device__ __forceinline__ static value_t device_cooperative_lookup(const device_handle_t<true>& handle_inst, const bool active, const key_t key) {
        static const typename IndexConfiguration::cooperative_search_algorithm_type cooperative_search_algorithm{};
        const auto pos = cooperative_search_algorithm(active, key, handle_inst.d_column, handle_inst.d_size);
        return (active && pos < handle_inst.d_size) ? static_cast<value_t>(pos) : invalid_tid;
    }

    __host__ void construct(const vector_view<key_t>& h_column, const key_t* d_column) override {
        _device_handle_inst.d_column = d_column;
        _device_handle_inst.d_size = h_column.size();
        if (_device_handle_inst.d_size >= invalid_tid) {
            throw std::runtime_error("'value_t' does not suffice");
        }
    }

    __host__ size_t memory_consumption() const override {
        return 0;
    }

    __host__ device_handle_t<true>& get_device_handle() {
        //return _device_handle_inst;
        auto handle = reinterpret_cast<device_handle_t<true>*>(&_device_handle_inst);
        return *handle;
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

    template<bool IsConst = false>
    struct device_handle_t {
        //const key_t* d_column;
        add_const_if_t<const key_t* __restrict__, IsConst> d_column;
        //device_size_t d_size;
        add_const_if_t<device_size_t, IsConst> d_size;

        /*
        [[deprecated]]
        __device__ __forceinline__ value_t lookup(const key_t key) const { return invalid_tid; }
        */
    };
    device_handle_t<false> _device_handle_inst;

    __device__ __forceinline__ static value_t device_cooperative_lookup(const device_handle_t<true>& handle_inst, const bool active, const key_t key) {
        return invalid_tid;
    }

    __host__ void construct(const vector_view<key_t>& /*h_column*/, const key_t* /*d_column*/) override {}

    __host__ size_t memory_consumption() const override {
        return 0;
    }

    __host__ device_handle_t<true>& get_device_handle() {
        //return _device_handle_inst;
        auto handle = reinterpret_cast<device_handle_t<true>*>(&_device_handle_inst);
        return *handle;
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
