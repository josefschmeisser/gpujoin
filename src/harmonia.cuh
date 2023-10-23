#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <random>
#include <iostream>
#include <functional>

#include <cub/util_debug.cuh>
#undef _Float16

#include "utils.hpp"
#include "cuda_utils.cuh"
#include "device_array.hpp"
#include "device_definitions.hpp"
#include "limited_vector.hpp"

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

#ifndef NRDC
#define HARMONIA_EXTERN_CACHE extern
#else
#define HARMONIA_EXTERN_CACHE
#endif

namespace harmonia {

//using child_ref_t = uint32_t; // TODO use device_size_t ?
using child_ref_t = device_size_t;

// only contains the upper tree levels; stored in constant memory
// retain some space for kernel launch arguments (those are also stored in constant memory)
static constexpr int harmonia_max_constant_mem = 42*1024;
HARMONIA_EXTERN_CACHE __constant__ child_ref_t harmonia_upper_levels[harmonia_max_constant_mem/sizeof(child_ref_t)];

template<
    class Key,
    class Value,
    template<class T> class HostAllocator,
    unsigned fanout,
    Value not_found,
    bool Sorted_Only = true>
struct harmonia_tree {
    using key_t = Key;
    using value_t = Value;

    static constexpr size_t max_depth = 16;
    static constexpr key_t max_key = std::numeric_limits<key_t>::max();
    static_assert(fanout < std::numeric_limits<unsigned>::max());
    static constexpr size_t max_keys = fanout - 1; // per node

    static size_t constexpr get_max_keys() noexcept { return max_keys; }

    limited_vector<key_t, HostAllocator<key_t>> keys;
    const key_t* leaf_keys;
    size_t key_count_prefix_sum;
    limited_vector<child_ref_t, HostAllocator<child_ref_t>> children;
    limited_vector<value_t, HostAllocator<value_t>> values;

    device_size_t size;
    unsigned depth;
    std::vector<unsigned> ntg_degrees;

    struct device_handle_t {
        const key_t* __restrict__ keys;
        const key_t* __restrict__ leaf_keys;
        const child_ref_t* __restrict__ children;
        const value_t* __restrict__ values;
        device_size_t size;
        device_size_t key_count_prefix_sum;
        unsigned depth;
        unsigned caching_depth;
        unsigned ntg_degree[max_depth]; // ntg size for each level starting at the root level
    };

    struct memory_guard_t {
        device_array_wrapper<key_t> keys_guard;
        device_array_wrapper<key_t> leaf_keys_guard;
        device_array_wrapper<child_ref_t> children_guard;
        device_array_wrapper<value_t> values_guard;
    };

#if 0
    struct node_description {
        //unsigned level = 0;
        bool is_leaf = false;
        size_t count = 0;
        size_t key_start_range = 0; // TODO remove
        //size_t key_end_range = 0;
        // either children offsets or value offsets
        //size_t payload_start_range = 0;
        child_ref_t child_ref_start = 0; // TODO remove
        //size_t payload_end_range = 0;
        //key_t keys[max_keys];
        /*
        union {
            node_description* children[fanout];
            value_t values[max_keys];
        };
        */
    };
#endif

    struct level_data {
        bool is_leaf_level = false;
        size_t key_count = 0;
        size_t node_count = 0;

        // cummulative members
        size_t node_count_prefix_sum = 0; // exclusive prefix sum
        size_t key_count_prefix_sum = 0; // exclusive prefix sum
    };

    template<class VectorType>
    __host__ std::vector<level_data> gather_tree_info(const VectorType& input) {
        std::vector<level_data> levels;
        levels.reserve(6);

        // in the first phase we count the pages on each level
        size_t previous_page_count = 0;
        size_t current_key_count = input.size();
        while (true) {
            //level_data& current_level_info = levels.emplace_back({});
            //levels.emplace_back({});
            levels.emplace(levels.begin());
            //level_data& current_level_info = levels.back();
            level_data& current_level_info = levels.front();
            current_level_info.key_count = current_key_count;

            size_t page_count = gather_level_info(current_level_info);
            assert(page_count > 0);
            // handle an edge case where the rightmost page on a level is empty, yet refers to a single child
            page_count += (fanout * page_count < previous_page_count) ? 1 : 0;

            //current_level_info.nodes.reserve(page_count);
            current_level_info.node_count = page_count;
            previous_page_count = page_count;

            //std::cout << "level current_key_count: " << current_key_count << " page_count: " << page_count << std::endl;
            current_key_count = std::floor(static_cast<float>(max_keys * page_count) / static_cast<float>(fanout));
            //std::cout << "next key count: " << current_key_count << std::endl;
            if (page_count == 1) {
                break;
            }
        }

        // mark leaf level
        levels.back().is_leaf_level = true;

        // prefix sum of all offsets over the levels in downward order
        size_t total_node_count = 0;
        size_t total_key_count = 0;
        for (auto& level : levels) {
            level.node_count_prefix_sum = total_node_count;
            total_node_count += level.node_count;

            level.key_count_prefix_sum = total_key_count;
            total_key_count += level.key_count;
        }

        return levels;
    }

    __host__ size_t gather_level_info(const level_data& l) {
        const auto page_count = (l.key_count + max_keys - 1) / max_keys;
        return page_count;
    }

#if 0
    __host__ void create_node_descriptions(std::vector<level_data>& levels) {
        unsigned level_index = 0;
        size_t current_key_index = 0;
        size_t current_children_index = 0;
        size_t current_value_index = 0;
        for (auto& level : levels) {
            const bool is_leaf_level = (level_index == levels.size() - 1);

            size_t keys_remaining = level.key_count;
            while (keys_remaining > 0) {
                level.nodes.emplace_back();
                node_description& node = level.nodes.back();

                node.is_leaf = is_leaf_level;
                node.count = std::min<size_t>(max_keys, keys_remaining);
                node.key_start_range = current_key_index;
                current_key_index += node.count;

                keys_remaining -= node.count;
            }

            level_index += 1;
        }
    }
#endif

    __host__ key_t get_largest_key(const std::vector<level_data>& tree_levels, unsigned level_idx, size_t node_idx) {
        assert(tree_levels.size() > level_idx);

        const auto& level_data = tree_levels[level_idx];
        const auto node_key_count = (node_idx < level_data.node_count - 1) ? max_keys : (level_data.key_count - node_idx*max_keys);

        if (tree_levels.size() - 1 == level_idx) {
            const size_t keys_start = max_keys*(level_data.node_count_prefix_sum + node_idx);
            return keys[keys_start + node_key_count - 1];
        }

        const size_t child_node_idx = fanout*node_idx + node_key_count;
        return get_largest_key(tree_levels, level_idx + 1, child_node_idx);
    }

    __host__ void populate_inner_nodes(const std::vector<level_data>& tree_levels, const size_t current_level) {
        //std::cout << "populate_inner_nodes level: " << current_level << std::endl;
        const auto& current_level_data = tree_levels.at(current_level);
        const auto& lower_level_data = tree_levels.at(current_level + 1);
        assert(lower_level_data.node_count > 1);

        const size_t parent_level_node_count = (current_level > 0) ? current_level_data.node_count_prefix_sum : 0;
        const size_t level_keys_start = parent_level_node_count * max_keys;

        // populate key array
        bool skipped = true;
        size_t next_key_idx = 0;
        for (size_t node_idx = 0; node_idx < lower_level_data.node_count - 1; ++node_idx) {
            // The separator between two pages is implicitly given by their parent node.
            // Hence, every `max_keys` keys one key can be omitted.
            if (!skipped && (next_key_idx % max_keys) == 0) {
                skipped = true;
                continue;
            }

            key_t sep = get_largest_key(tree_levels, current_level + 1, node_idx);
            //std::cout << "sep: " << sep << std::endl;
            keys[level_keys_start + next_key_idx] = sep;
            next_key_idx += 1;
            skipped = false;
        }

        if (current_level == 0) return;

        // ascend to next level
        populate_inner_nodes(tree_levels, current_level - 1);
    }

    template<class Vector>
    __host__ void populate_leaf_nodes(const std::vector<level_data>& tree_levels, const Vector& input) {
        size_t children_offset = tree_levels.back().node_count_prefix_sum;
        //size_t key_count_prefix_sum = children_offset * max_keys;
        key_count_prefix_sum = children_offset * max_keys;
#if 1
#else
        // TODO remove:
        std::copy(input.begin(), input.end(), keys.begin() + key_count_prefix_sum);
#endif

        child_ref_t values_prefix_sum = 0;
        for (size_t node_idx = 0; node_idx < tree_levels.back().node_count; ++node_idx) {
            children[children_offset] = values_prefix_sum;

            // prepare next iteration
            values_prefix_sum += max_keys;
            children_offset += 1;
        }
    }

    template<class Vector>
    __host__ void populate_nodes(const std::vector<level_data>& tree_levels, const Vector& input) {
        printf("before populate_leaf_nodes\n");
        populate_leaf_nodes(tree_levels, input);
        printf("after populate_leaf_nodes\n");
        if (tree_levels.size() > 1) {
            populate_inner_nodes(tree_levels, tree_levels.size() - 2);
        }
    }

    __host__ void store_structure(const std::vector<level_data>& tree_levels) {
        size_t children_offset = 0;
        child_ref_t prefix_sum = 1;

        // levels are stored in reverse order since the tree was constructed in bottom-up fashion
        for (const auto level : tree_levels) {
            if (level.is_leaf_level) break;

            // write out the prefix sum array entries
            for (size_t node_idx = 0; node_idx < level.node_count; ++node_idx) {
                children[children_offset++] = prefix_sum;

                const auto node_key_count = (node_idx < level.node_count - 1) ? max_keys : (level.key_count - node_idx*max_keys);
                //std::cout << "node_key_count: " << node_key_count << " l k count: " << level.key_count << " node_idx: " << node_idx << " c: " << level.key_count - node_idx*max_keys << std::endl;

                prefix_sum += node_key_count + 1;
            }
        }
    }

    template<class Vector>
    __host__ void construct(const Vector& input) {
        assert(input.size() < std::numeric_limits<device_size_t>::max());

leaf_keys = input.data();
        auto levels = gather_tree_info(input);
        //create_node_descriptions(levels);

        // allocate memory for all the arrays
        const auto& leaf_level = levels.back();
        const auto node_count = leaf_level.node_count_prefix_sum + leaf_level.node_count;
        const auto key_array_size = max_keys*node_count;

#if 1

        decltype(keys) new_keys(key_array_size, key_array_size - leaf_level.node_count);
#else
        decltype(keys) new_keys(key_array_size, key_array_size);
#endif
        keys.swap(new_keys);
        decltype(children) new_children(node_count, node_count);
        children.swap(new_children);
        if /*constexpr*/ (!Sorted_Only) {
            printf("=== sorted only!\n");
            decltype(values) new_values(input.size(), input.size());
            values.swap(new_values);
        }

        printf("fill with max_key\n");
        // populate the entire key array with `max_key`
        // so that underfull nodes do not require any special logic during lookup
        std::fill(keys.begin(), keys.end(), key_t(max_key));

        // initialize remaining members
        depth = levels.size();
        size = input.size();

        printf("invoke store_structure\n");
        store_structure(levels);
        printf("invoke populate_nodes\n");
        populate_nodes(levels, input);
        printf("structure complete\n");

        if (depth > max_depth) throw std::runtime_error("max depth exceeded");

        if (input.size() < 1000*10) return;

        // optimize ntg sizes
        std::vector<key_t> key_sample;
        key_sample.reserve(1000);
        // sample was introduced in c++17
        //std::sample(input.begin(), input.end(), std::back_inserter(key_sample), 1000, std::mt19937{std::random_device{}()});
        printf("before simple_sample\n");
        simple_sample(input.begin(), input.end(), std::back_inserter(key_sample), 1000, std::mt19937{std::random_device{}()});
        printf("after simple_sample\n");
        printf("before optimize ntg\n");
        //optimize_ntg(key_sample);
        printf("after optimize ntg\n");
    }

    // return: {depth_limit, byte_limit}
    __host__ std::pair<unsigned, size_t> determine_children_caching_limit(size_t available_bytes) {
        constexpr key_t largest_key = std::numeric_limits<key_t>::max() - 1;

        unsigned current_depth = 0;
        size_t lb = 0, pos = 0;
        for (; current_depth < depth; ++current_depth) {
            const key_t* node_start = &keys[max_keys*pos];

            lb = std::lower_bound(node_start, node_start + max_keys, largest_key) - node_start;
            size_t new_pos = children[pos] + lb;
            if (new_pos*sizeof(child_ref_t) > available_bytes) {
                break;
            }

            pos = new_pos;
        }

        return {current_depth, pos*sizeof(child_ref_t)};
    }

    __host__ unsigned copy_children_portion_to_cached_memory() {
        const size_t available_bytes = sizeof(harmonia_upper_levels);
        printf("harmonia accessible memory: %lu\n", available_bytes);

        //const auto [resulting_caching_depth, bytes_to_copy] = determine_children_caching_limit(available_bytes);
        unsigned caching_depth;
        size_t bytes_to_copy;
        std::tie(caching_depth, bytes_to_copy) = determine_children_caching_limit(available_bytes);

        //caching_depth = resulting_caching_depth;
        CubDebugExit(cudaMemcpyToSymbol(harmonia_upper_levels, children.data(), bytes_to_copy));

        printf("harmonia constant memory required: %lu depth limit: %u full depth: %u\n", bytes_to_copy, caching_depth, depth);
        return caching_depth;
    }

    // TODO remove
    __host__ void create_device_handle(device_handle_t& handle) {
        // copy upper tree levels to device constant memory
        // TODO re-enable
        //const auto caching_depth = copy_children_portion_to_cached_memory();
        const auto caching_depth = 0;

        key_t* d_keys;
        auto ret = cudaMalloc(&d_keys, sizeof(key_t)*keys.size());
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(d_keys, keys.data(), sizeof(key_t)*keys.size(), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);

        child_ref_t* d_children;
        ret = cudaMalloc(&d_children, sizeof(child_ref_t)*children.size());
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(d_children, children.data(), sizeof(child_ref_t)*children.size(), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);

        value_t* d_values = nullptr;
        if /*constexpr*/ (!Sorted_Only) {
            ret = cudaMalloc(&d_values, sizeof(value_t)*values.size());
            assert(ret == cudaSuccess);
            ret = cudaMemcpy(d_values, values.data(), sizeof(value_t)*values.size(), cudaMemcpyHostToDevice);
            assert(ret == cudaSuccess);
        }

        // initialize fields
        handle.depth = depth;
        handle.caching_depth = caching_depth;
        handle.size = size;
        handle.keys = d_keys;
        handle.children = d_children;
        handle.values = d_values;

        // copy ntg degrees to device accessible array
        assert(ntg_degrees.size() < max_depth);
        for (unsigned i = 0; i < ntg_degrees.size(); ++i) {
            handle.ntg_degree[i] = ntg_degrees[i];
        }
    }

    template<class DeviceAllocator>
    __host__ void create_device_handle(device_handle_t& handle, DeviceAllocator& device_allocator, memory_guard_t& guard) {
        printf("create_device_handle\n");
        // TODO re-enable
        //const auto caching_depth = copy_children_portion_to_cached_memory();
        const auto caching_depth = 0;

        // initialize fields
        handle.depth = depth;
        handle.caching_depth = caching_depth;
        handle.size = size;

        // migrate key array
        typename DeviceAllocator::rebind<key_t>::other keys_allocator = device_allocator;
        guard.keys_guard = create_device_array_from(keys, keys_allocator);
        handle.keys = guard.keys_guard.data();

        // migrate leaf key array
//guard.leaf_keys_guard = create_device_array_from(const_cast<key_t*>(leaf_keys), keys_allocator);
//handle.leaf_keys = guard.leaf_keys_guard.data();
handle.leaf_keys = leaf_keys;

handle.key_count_prefix_sum = key_count_prefix_sum;

        // migrate children array
        typename DeviceAllocator::rebind<child_ref_t>::other children_allocator = device_allocator;
        guard.children_guard = create_device_array_from(children, children_allocator);
        handle.children = guard.children_guard.data();

        if /*constexpr*/ (!Sorted_Only) {
            typename DeviceAllocator::rebind<value_t>::other values_allocator = device_allocator;
            guard.values_guard = create_device_array_from(values, values_allocator);
            handle.values = guard.values_guard.data();
        }

        // copy ntg degrees to device accessible array
        assert(ntg_degrees.size() < max_depth);
        for (unsigned i = 0; i < ntg_degrees.size(); ++i) {
            handle.ntg_degree[i] = ntg_degrees[i];
        }
    }

    // utilize the full warp for each query
    __device__ static device_size_t cooperative_linear_search(const bool active, const key_t x, const key_t* arr) {
        device_size_t lower_bound = max_keys;
        const unsigned my_lane_id = lane_id();
        unsigned leader = 0;
        const int lane_offset = my_lane_id - leader;
        assert(my_lane_id >= leader);

        // iterate over all threads within a cooperative group by shifting the leader thread from the lsb to the msb within the window mask
        for (unsigned shift = 0; shift < 32; ++shift) {
            int key_idx = lane_offset - 32;
            const key_t leader_x = __shfl_sync(FULL_MASK, x, leader);
            const key_t* leader_arr = reinterpret_cast<const key_t*>(__shfl_sync(FULL_MASK, reinterpret_cast<uint64_t>(arr), leader));

            const auto leader_active = __shfl_sync(FULL_MASK, active, leader);
            bool advance = leader_active;
            uint32_t matches = 0;
            while (matches == 0 && advance) {
                key_idx += 32;

                key_t value;
                if (key_idx < max_keys) value = leader_arr[key_idx];
                matches = __ballot_sync(FULL_MASK, key_idx < max_keys && value >= leader_x);
                advance = key_idx - lane_offset + 32 < max_keys; // termination criterion
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
    }

    template<unsigned Degree>
    __device__ static device_size_t cooperative_linear_search(const bool active, const key_t x, const key_t* arr) {
        enum { WINDOW_SIZE = 1 << Degree };

        device_size_t lower_bound = max_keys;
        const unsigned my_lane_id = lane_id();
        unsigned leader = WINDOW_SIZE*(my_lane_id >> Degree); // equivalent to WINDOW_SIZE*(my_lane_id div WINDOW_SIZE)
        const uint32_t window_mask = ((1u << WINDOW_SIZE) - 1u) << leader;
        assert(my_lane_id >= leader);
        const int lane_offset = my_lane_id - leader;

        // iterate over all threads within a cooperative group by shifting the leader thread from the lsb to the msb within the window mask
        for (unsigned shift = 0; shift < WINDOW_SIZE; ++shift) {
            int key_idx = lane_offset - WINDOW_SIZE;
            const key_t leader_x = __shfl_sync(window_mask, x, leader);
            const key_t* leader_arr = reinterpret_cast<const key_t*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));

            const auto leader_active = __shfl_sync(window_mask, active, leader);
            bool advance = leader_active;
            uint32_t matches = 0;
            while (matches == 0 && advance) {
                key_idx += WINDOW_SIZE;

                key_t value;
                if (key_idx < max_keys) value = leader_arr[key_idx];
                //const key_t value = leader_arr[(key_idx < max_keys) ? key_idx : 0]; // almost 10 percent slower than the conditional version
                matches = __ballot_sync(window_mask, key_idx < max_keys && value >= leader_x);
                advance = key_idx - lane_offset + WINDOW_SIZE < max_keys; // termination criterion
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
    }

    __device__ static device_size_t cooperative_linear_search(const bool active, const key_t x, const key_t* arr, const unsigned ntg_degree) {
        device_size_t lower_bound = max_keys;
        const unsigned my_lane_id = lane_id();
        const unsigned ntg_size = 1u << ntg_degree;
        unsigned leader = ntg_size*(my_lane_id >> ntg_degree); // equivalent to ntg_size*(my_lane_id div ntg_size)
        const int lane_offset = my_lane_id - leader;
        const uint32_t window_mask = __funnelshift_l(FULL_MASK, 0, ntg_size) << leader;
        assert(my_lane_id >= leader);

        // iterate over all threads within a cooperative group by shifting the leader thread from the lsb to the msb within the window mask
        for (unsigned shift = 0; shift < ntg_size; ++shift) {
            int key_idx = lane_offset - ntg_size;
            const key_t leader_x = __shfl_sync(window_mask, x, leader);
            const key_t* leader_arr = reinterpret_cast<const key_t*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));

            const auto leader_active = __shfl_sync(window_mask, active, leader);
            bool advance = leader_active;
            uint32_t matches = 0;
            while (matches == 0 && advance) {
                key_idx += ntg_size;

                key_t value;
                // TODO check if omitting this if statement while increasing the key array size accordingly leads to any performance difference
                if (key_idx < max_keys) value = leader_arr[key_idx];
                matches = __ballot_sync(window_mask, key_idx < max_keys && value >= leader_x);
                advance = key_idx - lane_offset + ntg_size < max_keys; // termination criterion
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
    }

    // host-side lookup function, for validation purposes only
    __host__ value_t lookup(key_t key) {
        bool active = true;
        key_t actual;
        device_size_t lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < depth; ++current_depth) {
            const key_t* node_start = &keys[max_keys*pos];

            lb = std::lower_bound(node_start, node_start + max_keys, key) - node_start;
            actual = node_start[lb];

            device_size_t new_pos = children[pos] + lb;

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < size && key == actual) {
            if /*constexpr*/ (Sorted_Only) {
                return pos;
            } else {
                return values[pos];
            }
        }

        return not_found;
    }

    template<unsigned Degree = 2> // cooperative parallelization Degree
    // This has to be a function template so that it won't get compiled when Sorted_Only is false.
    // To make it a function template, we have to add the second predicate to std::enable_if_t which is dependent on the function template parameter.
    // And with the help of SFINAE only the correct implementation will get compiled.
    __device__ static std::enable_if_t<Sorted_Only && Degree < 6, value_t> lookup(const bool active, const device_handle_t& tree, const key_t key) {
#ifndef NDEBUG
        __syncwarp();
        assert(__activemask() == FULL_MASK); // ensure that all threads participate
#endif
        key_t actual;
        device_size_t lb = 0, pos = 0;
        for (unsigned current_depth = 1; current_depth <= tree.depth; ++current_depth) {


            assert(!active || current_depth < tree.depth || tree.key_count_prefix_sum <= max_keys*pos);
            //const key_t* keys = current_depth == tree.depth ? (tree.leaf_keys - max_keys*pos) : tree.keys;
            const key_t* keys = current_depth == tree.depth ? (tree.leaf_keys - tree.key_count_prefix_sum) : tree.keys;
            //const key_t* keys = tree.keys;
            const key_t* node_start = keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search<Degree>(active, key, node_start);
            actual = node_start[lb];

            device_size_t new_pos = tree.children[pos] + lb;
            //active = active && new_pos < tree.size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree.size && key == actual) {
            return pos;
        }

        return not_found;
    }

    template<unsigned Degree = 2>
    __device__ static std::enable_if_t<!Sorted_Only && Degree < 6, value_t> lookup(bool active, const device_handle_t& tree, key_t key) {
#ifndef NDEBUG
        __syncwarp();
        assert(__activemask() == FULL_MASK); // ensure that all threads participate
#endif
        key_t actual;
        device_size_t lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree.depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search<Degree>(active, key, node_start);
            actual = node_start[lb];

            device_size_t new_pos = tree.children[pos] + lb;
            //active = active && new_pos < tree.size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree.size && key == actual) {
            return tree.values[pos];
        }

        return not_found;
    }

    __device__ static value_t ntg_lookup(const bool active, const device_handle_t& tree, const key_t key) {
#ifndef NDEBUG
        __syncwarp();
        assert(__activemask() == FULL_MASK); // ensure that all threads participate
#endif
        key_t actual;
        device_size_t lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree.depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search(active, key, node_start, tree.ntg_degree[current_depth]);
            actual = node_start[lb];

            device_size_t new_pos = tree.children[pos] + lb;

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree.size && key == actual) {
            return pos; // FIXME: use a compile time switch
        }

        return not_found;
    }

    __device__ static value_t ntg_lookup_with_caching(const bool active, const device_handle_t& tree, const key_t key) {
#ifndef NDEBUG
        __syncwarp();
        assert(__activemask() == FULL_MASK); // ensure that all threads participate
#endif
        key_t actual;
        device_size_t lb = 0, pos = 0, current_depth = 0;
        // use the portion of the children array which is stored in constant memory; hence, accesses to this array will be cached
        for (; current_depth < tree.caching_depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search(active, key, node_start, tree.ntg_degree[current_depth]);
            actual = node_start[lb];

            device_size_t new_pos = harmonia_upper_levels[pos] + lb;
            //active = active && new_pos < tree.size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }
        // once the upper portion of the children array is exhausted, we switch to the full array which is kept in global memory
        for (; current_depth < tree.depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search(active, key, node_start, tree.ntg_degree[current_depth]);
            actual = node_start[lb];

            device_size_t new_pos = tree.children[pos] + lb;
            //active = active && new_pos < tree.size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree.size && key == actual) {
            //return tree.values[pos];
            return pos; // FIXME: use a compile time switch
        }

        return not_found;
    }

    __host__ unsigned count_ntg_steps(const key_t x, const key_t* arr, const unsigned ntg_degree) {
        const unsigned ntg_size = 1u << ntg_degree;
        //std::cout << "ntg_degree: " << ntg_degree << " ntg_size: " << ntg_size << std::endl;
        const device_size_t lb = std::lower_bound(arr, arr + max_keys, x) - arr;
        //std::cout << "lb: " << lb << std::endl;
        return 1 + std::min<device_size_t>(lb, max_keys - 1)/ntg_size;
    }

    // return: the number of ntg windows shifts required
    __host__ unsigned count_ntg_steps_at_target_depth(const unsigned target_depth, const unsigned current_ntg_degree, const key_t key) {
        // traverse upper levels
        device_size_t lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < target_depth; ++current_depth) {
            const key_t* node_start = &keys[max_keys*pos];
            lb = std::lower_bound(node_start, node_start + max_keys, key) - node_start;
            device_size_t new_pos = children[pos] + lb;
            pos = new_pos;
        }
 
        // reached the target depth; count the ntg window shifts 
        const key_t* node_start = &keys[max_keys*pos];
        const auto steps = count_ntg_steps(key, node_start, current_ntg_degree);
        //std::cout << "steps: " << steps << " key: " << key << std::endl;
        return steps;
    }

    __host__ void optimize_ntg(std::vector<key_t>& sample) {
        printf("in optimize_ntg\n");
        ntg_degrees.clear();
        for (unsigned current_depth = 0; current_depth < depth; ++current_depth) {
            unsigned current_ntg_degree = 5;
            double avg_steps_before, avg_steps_after;
            uint64_t acc_steps = 0;
            for (const key_t key : sample) {
                acc_steps += count_ntg_steps_at_target_depth(current_depth, current_ntg_degree, key);
            }
            avg_steps_before = static_cast<double>(acc_steps)/sample.size();

            std::cout << "depth " << current_depth << " avg_steps_before: " << avg_steps_before << std::endl;

            // narrow the thread group
            double factor;
            do {
                current_ntg_degree -= 1;
                acc_steps = 0;
                for (const key_t key : sample) {
                    acc_steps += count_ntg_steps_at_target_depth(current_depth, current_ntg_degree, key);
                }
                avg_steps_after = static_cast<double>(acc_steps)/sample.size();
                std::cout << "depth " << current_depth << " avg_steps_after : " << avg_steps_after << std::endl;

                factor = 2.*avg_steps_before/avg_steps_after;
                std::cout << "factor: " << factor << std::endl;
                avg_steps_before = avg_steps_after;
            } while (factor > 1. && current_ntg_degree > 1);
            current_ntg_degree += 1; // the final narrowing does not improve the throughput

            std::cout << "=== depth " << current_depth << " final ntg_degree: " << current_ntg_degree << std::endl;
            ntg_degrees.push_back(current_ntg_degree);
        }
    }
};

}
