#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <random>
#include <iostream>

#include <cub/util_debug.cuh>

#include "utils.hpp"
#include "cuda_utils.cuh"
#include "device_array.hpp"

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

#ifndef NRDC
#define HARMONIA_EXTERN_CACHE extern
#else
#define HARMONIA_EXTERN_CACHE
#endif

namespace harmonia {

using child_ref_t = uint32_t;

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

    static const unsigned max_depth = 16;
    static constexpr auto max_keys = fanout - 1;

    std::vector<key_t, HostAllocator<key_t>> keys;
    std::vector<child_ref_t, HostAllocator<child_ref_t>> children;
    std::vector<value_t, HostAllocator<value_t>> values;

    unsigned size;
    unsigned depth;
    std::vector<unsigned> ntg_degrees;

    struct device_handle_t {
        const key_t* __restrict__ keys;
        const child_ref_t* __restrict__ children;
        const value_t* __restrict__ values;
        unsigned size;
        unsigned depth;
        unsigned caching_depth;
        unsigned ntg_degree[max_depth]; // ntg size for each level starting at the root level
    };

    struct memory_guard_t {
        device_array_wrapper<key_t> keys_guard;
        device_array_wrapper<child_ref_t> children_guard;
        device_array_wrapper<value_t> values_guard;
    };

    struct intermediate_node {
        bool is_leaf = false;
        unsigned count = 0;
        key_t keys[max_keys];
        union {
            intermediate_node* children[fanout];
            value_t values[max_keys];
        };
    };

    using tree_level_t = std::vector<std::unique_ptr<intermediate_node>>;
    using tree_levels_t = std::vector<std::unique_ptr<tree_level_t>>;

    __host__ key_t max_key(const intermediate_node& subtree) {
        if (subtree.is_leaf) {
            return subtree.keys[subtree.count - 1];
        }
        return max_key(*subtree.children[subtree.count]);
    }

    __host__ void add(intermediate_node& node, key_t key, value_t value) {
        node.keys[node.count] = key;
        node.values[node.count] = value;
        node.count += 1;
    }

    __host__ void add(intermediate_node& node, key_t key, intermediate_node* child) {
        node.keys[node.count] = key;
        node.children[node.count] = child;
        node.count += 1;
    }

    __host__ void construct_inner_nodes(tree_levels_t& tree_levels) {
        const auto& lower_level = tree_levels.back();

        if (lower_level->size() == 1) {
            return;
        }

        auto current_level = std::make_unique<tree_level_t>();
        auto node = std::make_unique<intermediate_node>();
        node->is_leaf = false;
        node->keys[0] = std::numeric_limits<key_t>::max();

        for (unsigned i = 0; i < lower_level->size() - 1; i++) {
            auto& curr = lower_level->at(i);
            if (node->count >= max_keys) {
                node->children[node->count] = curr.get();
                current_level->push_back(std::move(node));
                node = std::make_unique<intermediate_node>();
                node->is_leaf = false;
                node->keys[0] = std::numeric_limits<key_t>::max();
            } else {
                key_t sep = max_key(*curr);
                add(*node, sep, curr.get());
            }
        }
        node->children[node->count] = lower_level->back().get();
        current_level->push_back(std::move(node));

        tree_levels.push_back(std::move(current_level));

        construct_inner_nodes(tree_levels);
    }

    template<class Vector>
    __host__ tree_levels_t construct_levels(const Vector& input) {
        uint64_t n = input.size();

        auto leaves = std::make_unique<tree_level_t>();
        auto node = std::make_unique<intermediate_node>();
        node->is_leaf = true;

        // construct leaves
        for (uint64_t i = 0; i < n; i++) {
            auto k = input[i];
            value_t value = i;
            if (node->count >= max_keys) {
                leaves->push_back(std::move(node));
                node = std::make_unique<intermediate_node>();
                node->is_leaf = true;
            }
            add(*node, k, value);
        }
        leaves->push_back(std::move(node));

        tree_levels_t tree_levels;
        tree_levels.push_back(std::move(leaves));

        // recursively construct the remaining levels in bottom-up fashion
        construct_inner_nodes(tree_levels);

        return tree_levels;
    }

    __host__ void fill_underfull_node(intermediate_node& node) {
        for (unsigned i = 1; i < max_keys; ++i) {
            if (node.keys[i - 1] > node.keys[i]) {
                node.keys[i] = std::numeric_limits<key_t>::max();
            }
        }
    }

    __host__ void store_nodes(tree_levels_t& tree_levels) {
        unsigned key_offset = 0;

        // the keys are stored in breadth first order
        for (auto it = std::rbegin(tree_levels); it != std::rend(tree_levels); ++it) {
            auto& tree_level = *(*it);
            fill_underfull_node(*tree_level.back());

            for (auto& node : tree_level) {
                std::memcpy(&keys[key_offset], node->keys, sizeof(key_t)*max_keys);
                key_offset += max_keys;
            }
        }
    }

    __host__ void store_structure(const tree_levels_t& tree_levels) {
        unsigned children_offset = 0;
        child_ref_t prefix_sum = 1;

        // levels are stored in reverse order since the tree was constructed in bottom-up fashion
        for (auto it = std::rbegin(tree_levels); it != std::rend(tree_levels); ++it) {
            auto& tree_level = *(*it);
            if (tree_level.front()->is_leaf) {
                child_ref_t values_prefix_sum = 0;
                for (auto& node : tree_level) {
                    children[children_offset++] = values_prefix_sum;
                    values_prefix_sum += max_keys;
                }
            } else {
                // write out the prefix sum array entries
                for (auto& node : tree_level) {
                    children[children_offset++] = prefix_sum;
                    prefix_sum += node->count + 1;
                }
            }
        }
    }

    template<class Vector>
    __host__ void construct(const Vector& input) {
        auto tree_levels = construct_levels(input);
        auto& root = tree_levels.front()->front();

        // allocate arrays
        // transform_reduce was introduced with c++17
        //const auto node_count = std::transform_reduce(tree_levels.begin(), tree_levels.end(), 0, std::plus<>(), [](auto& level) { return level->size(); });
        unsigned node_count = 0;
        for (const auto& level : tree_levels) {
            node_count += level->size();
        }

        const auto key_array_size = max_keys*node_count;
        keys.resize(key_array_size);
        children.resize(node_count);
        if /*constexpr*/ (!Sorted_Only) {
            values.resize(input.size());
        }

        store_nodes(tree_levels);
        store_structure(tree_levels);

        if /*constexpr*/ (!Sorted_Only) {
            // insert values
            for (unsigned i = 0; i < input.size(); ++i) {
                values[i] = i;
            }
        }

        // initialize remaining members
        depth = tree_levels.size();
        size = input.size();

	if (depth > max_depth) throw std::runtime_error("max depth exceeded");

        if (input.size() < 1000*10) return;

        // optimize ntg sizes
        std::vector<key_t> key_sample;
        key_sample.reserve(1000);
        // sample was introduced with c++17
        //std::sample(input.begin(), input.end(), std::back_inserter(key_sample), 1000, std::mt19937{std::random_device{}()});
        simple_sample(input.begin(), input.end(), std::back_inserter(key_sample), 1000, std::mt19937{std::random_device{}()});
        optimize_ntg(key_sample);
    }

    // return: {depth_limit, byte_limit}
    __host__ std::pair<unsigned, size_t> determine_children_caching_limit(size_t available_bytes) {
        constexpr key_t largest_key = std::numeric_limits<key_t>::max() - 1;

        unsigned current_depth = 0;
        unsigned lb = 0, pos = 0;
        for (; current_depth < depth; ++current_depth) {
            const key_t* node_start = &keys[max_keys*pos];

            lb = std::lower_bound(node_start, node_start + max_keys, largest_key) - node_start;
            unsigned new_pos = children[pos] + lb;
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

    __host__ void create_device_handle(device_handle_t& handle) {
        // copy upper tree levels to device constant memory
        const auto caching_depth = copy_children_portion_to_cached_memory();

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
        // copy upper tree levels to device constant memory
        const auto caching_depth = copy_children_portion_to_cached_memory();

        // initialize fields
        handle.depth = depth;
        handle.caching_depth = caching_depth;
        handle.size = size;

        // migrate key array
        typename DeviceAllocator::rebind<key_t>::other keys_allocator = device_allocator;
        guard.keys_guard = create_device_array_from(keys, keys_allocator);
        handle.keys = guard.keys_guard.data();

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
    __device__ static unsigned cooperative_linear_search(const bool active, const key_t x, const key_t* arr) {
        unsigned lower_bound = max_keys;
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
    __device__ static value_t cooperative_linear_search(const bool active, const key_t x, const key_t* arr) {
        enum { WINDOW_SIZE = 1 << Degree };

        const unsigned my_lane_id = lane_id();
        unsigned leader = WINDOW_SIZE*(my_lane_id >> Degree); // equivalent to WINDOW_SIZE*(my_lane_id div WINDOW_SIZE)
        const uint32_t window_mask = ((1u << WINDOW_SIZE) - 1u) << leader;
        assert(my_lane_id >= leader);
        const int lane_offset = my_lane_id - leader;
        unsigned lower_bound = max_keys;

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

    __device__ static unsigned cooperative_linear_search(const bool active, const key_t x, const key_t* arr, const unsigned ntg_degree) {
        unsigned lower_bound = max_keys;
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
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < depth; ++current_depth) {
            const key_t* node_start = &keys[max_keys*pos];

            lb = std::lower_bound(node_start, node_start + max_keys, key) - node_start;
            actual = node_start[lb];

            unsigned new_pos = children[pos] + lb;

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
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree.depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search<Degree>(active, key, node_start);
            actual = node_start[lb];

            unsigned new_pos = tree.children[pos] + lb;
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
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree.depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search<Degree>(active, key, node_start);
            actual = node_start[lb];

            unsigned new_pos = tree.children[pos] + lb;
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
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree.depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search(active, key, node_start, tree.ntg_degree[current_depth]);
            actual = node_start[lb];

            unsigned new_pos = tree.children[pos] + lb;

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
        unsigned lb = 0, pos = 0, current_depth = 0; 
        // use the portion of the children array which is stored in constant memory; hence, accesses to this array will be cached
        for (; current_depth < tree.caching_depth; ++current_depth) {
            const key_t* node_start = tree.keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search(active, key, node_start, tree.ntg_degree[current_depth]);
            actual = node_start[lb];

            unsigned new_pos = harmonia_upper_levels[pos] + lb;
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

            unsigned new_pos = tree.children[pos] + lb;
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
        const unsigned lb = std::lower_bound(arr, arr + max_keys, x) - arr;
        //std::cout << "lb: " << lb << std::endl;
        return 1 + std::min(lb, max_keys - 1)/ntg_size;
    }

    // return: the number of ntg windows shifts required
    __host__ unsigned count_ntg_steps_at_target_depth(const unsigned target_depth, const unsigned current_ntg_degree, const key_t key) {
        // traverse upper levels
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < target_depth; ++current_depth) {
            const key_t* node_start = &keys[max_keys*pos];
            lb = std::lower_bound(node_start, node_start + max_keys, key) - node_start;
            unsigned new_pos = children[pos] + lb;
            pos = new_pos;
        }
 
        // reached the target depth; count the ntg window shifts 
        const key_t* node_start = &keys[max_keys*pos];
        const auto steps = count_ntg_steps(key, node_start, current_ntg_degree);
        //std::cout << "steps: " << steps << " key: " << key << std::endl;
        return steps;
    }

    __host__ void optimize_ntg(std::vector<key_t>& sample) {
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
