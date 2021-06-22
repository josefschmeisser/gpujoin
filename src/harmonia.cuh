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

#include "utils.hpp"
#include "cuda_utils.cuh"

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

namespace harmonia {

template<
    class Key,
    class Value,
    unsigned fanout,
    Value not_found,
    bool Sorted_Only = true>
struct harmonia_tree {
    using key_t = Key;
    using value_t = Value;
    using child_ref_t = uint32_t;

    static const unsigned max_depth = 16;
    static constexpr auto max_keys = fanout - 1;

    std::vector<key_t> keys;
    std::vector<child_ref_t> children;
    std::vector<value_t> values;
    unsigned size;
    unsigned depth;

    struct device_handle_t {
        const key_t* __restrict__ keys;
        const child_ref_t* __restrict__ children;
        const value_t* __restrict__ values;
        unsigned size;
        unsigned depth;
        unsigned ntg_degree[max_depth]; // ntg size for each level starting at the root level
    }* device_handle = nullptr;

/*
    struct device_handle_t {
        key_t* keys;
        child_ref_t* children;
        value_t* values;
        unsigned size;
        unsigned depth;
    }* device_handle = nullptr;
*/

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

    ~harmonia_tree() {
        // free device handle
        if (device_handle) {
// TODO
        }
    }

    key_t max_key(const intermediate_node& subtree) {
        if (subtree.is_leaf) {
            return subtree.keys[subtree.count - 1];
        }
        return max_key(*subtree.children[subtree.count]);
    }

    void add(intermediate_node& node, key_t key, value_t value) {
        node.keys[node.count] = key;
        node.values[node.count] = value;
        node.count += 1;
    }

    void add(intermediate_node& node, key_t key, intermediate_node* child) {
        node.keys[node.count] = key;
        node.children[node.count] = child;
        node.count += 1;
    }

    void construct_inner_nodes(tree_levels_t& tree_levels) {
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

    tree_levels_t construct_levels(const std::vector<key_t>& input) {
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

    void fill_underfull_node(intermediate_node& node) {
        for (unsigned i = 1; i < max_keys; ++i) {
            if (node.keys[i - 1] > node.keys[i]) {
                node.keys[i] = std::numeric_limits<key_t>::max();
            }
        }
    }

    void store_nodes(tree_levels_t& tree_levels) {
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

    void store_structure(const tree_levels_t& tree_levels) {
        unsigned children_offset = 0;
        child_ref_t prefix_sum = 1;

        // levels are stored in reverse order since the tree was constructed in bottom-up fashion
        for (auto it = std::rbegin(tree_levels); it != std::rend(tree_levels); ++it) {
            auto& tree_level = *(*it);
            if (tree_level.front()->is_leaf) {
                child_ref_t values_prefix_sum = 0;
                for (auto& node : tree_level) {
                    children[children_offset++] = values_prefix_sum;
                    values_prefix_sum += max_keys;// fanout;
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

    void construct(const std::vector<key_t>& input) {
        auto tree_levels = construct_levels(input);
        auto& root = tree_levels.front()->front();

        // allocate arrays
        const auto node_count = std::transform_reduce(tree_levels.begin(), tree_levels.end(), 0, std::plus<>(), [](auto& level) { return level->size(); });
        const auto key_array_size = max_keys*node_count;
        keys.resize(key_array_size);
        children.resize(node_count);
	if constexpr (!Sorted_Only) {
            values.resize(input.size());
	}

        store_nodes(tree_levels);
        store_structure(tree_levels);

        if constexpr (!Sorted_Only) {
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
        std::sample(input.begin(), input.end(), std::back_inserter(key_sample), 1000, std::mt19937{std::random_device{}()});
        optimize_ntg(key_sample);
    }

    void create_device_handle() {//device_handle_t& handle) {
        key_t* d_keys;
        auto ret = cudaMalloc(&d_keys, sizeof(key_t)*keys.size());
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(d_keys, keys.data(), sizeof(key_t)*keys.size(), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);

        child_ref_t* d_children;
        ret = cudaMalloc(&d_children, sizeof(child_ref_t)*children.size());
//printf("child array bytes: %u\n", sizeof(child_ref_t)*children.size());
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(d_children, children.data(), sizeof(key_t)*children.size(), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);
/* TODO
        if constexpr (!Sorted_Only) {
            ret = cudaMalloc(&tmp.values, sizeof(value_t)*values.size());
            assert(ret == cudaSuccess);
            ret = cudaMemcpy(tmp.values, values.data(), sizeof(key_t)*values.size(), cudaMemcpyHostToDevice);
            assert(ret == cudaSuccess);
        }
*/
        // initialize fields
        device_handle_t tmp;
        tmp.depth = depth;
        tmp.size = size;
        tmp.keys = d_keys;
        tmp.children = d_children;
        // TODO

	for (unsigned i = 0; i < max_depth; ++i) {
            tmp.ntg_degree[i] = 3;
	}

        // create cuda struct
        ret = cudaMalloc(&device_handle, sizeof(device_handle_t));
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(device_handle, &tmp, sizeof(device_handle_t), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);
    }

    template<unsigned Degree>
    __device__ static value_t cooperative_linear_search(const bool active, const key_t x, const key_t* arr) {
        enum { WINDOW_SIZE = 1 << Degree };

        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads within the warp participate

        const unsigned my_lane_id = lane_id();
        unsigned leader = WINDOW_SIZE*(my_lane_id >> Degree);
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
            unsigned exhausted_cnt = leader_active ? 0 : WINDOW_SIZE;
            uint32_t matches = 0;
            while (matches == 0 && exhausted_cnt < WINDOW_SIZE) {
                key_idx += WINDOW_SIZE;

                key_t value;
                if (key_idx < max_keys) value = leader_arr[key_idx];
                matches = __ballot_sync(window_mask, key_idx < max_keys && value >= leader_x);
                exhausted_cnt = __popc(__ballot_sync(window_mask, key_idx >= max_keys));
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
    }

    template<unsigned Degree = 2> // cooperative parallelization Degree
    // This has to be a function template so that it won't get compiled when Sorted_Only is false.
    // To make it a function template, we have to add the second predicate to std::enable_if_t which is dependent on the function template parameter.
    // And with the help of SFINAE only the correct implementation will get compiled.
    __device__ static std::enable_if_t<Sorted_Only && Degree < 6, value_t> lookup(const bool active, const device_handle_t* tree, const key_t key) {
        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        key_t actual;
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree->depth; ++current_depth) {
            const key_t* node_start = tree->keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search<Degree>(active, key, node_start);
            actual = node_start[lb];

            unsigned new_pos = tree->children[pos] + lb;
//            active = active && new_pos < tree->size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree->size && key == actual) {
            return pos;
        }

        return not_found;
    }

    template<unsigned Degree = 2>
    __device__ static std::enable_if_t<!Sorted_Only && Degree < 6, value_t> lookup(bool active, const device_handle_t* tree, key_t key) {
        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        key_t actual;
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree->depth; ++current_depth) {
            const key_t* node_start = tree->keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search<Degree>(active, key, node_start);
            actual = node_start[lb];

            unsigned new_pos = tree->children[pos] + lb;
//            active = active && new_pos < tree->size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree->size && key == actual) {
            return tree->values[pos];
        }

        return not_found;
    }

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
            if constexpr (Sorted_Only) {
                return pos;
            } else {
                return values[pos];
            }
        }

        return not_found;
    }

#if 1
    __device__ unsigned cooperative_linear_search(const bool active, const key_t x, const key_t* arr, const unsigned ntg_degree) {
        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        unsigned lower_bound = max_keys;
	const unsigned my_lane_id = lane_id();
	const unsigned ntg_size = 1u << ntg_degree;
        unsigned leader = ntg_size*(my_lane_id >> ntg_degree);
        const int lane_offset = my_lane_id - leader;
        const uint32_t window_mask = (ntg_size - 1u) << leader;
        assert(my_lane_id >= leader);

        // iterate over all threads within a cooperative group by shifting the leader thread from the lsb to the msb within the window mask
        for (unsigned shift = 0; shift < ntg_size; ++shift) {
            int key_idx = lane_offset - ntg_size;
            const key_t leader_x = __shfl_sync(window_mask, x, leader);
            const key_t* leader_arr = reinterpret_cast<const key_t*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));

            const auto leader_active = __shfl_sync(window_mask, active, leader);
            unsigned exhausted_cnt = leader_active ? 0 : ntg_size;
            uint32_t matches = 0;
            while (matches == 0 && exhausted_cnt < ntg_size) {
                key_idx += ntg_size;

                key_t value;
                if (key_idx < max_keys) value = leader_arr[key_idx];
                matches = __ballot_sync(window_mask, key_idx < max_keys && value >= leader_x);
                exhausted_cnt = __popc(__ballot_sync(window_mask, key_idx >= max_keys));
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
    }

#if 1
    __device__ static value_t ntg_lookup(const bool active, const device_handle_t* tree, const key_t key) {
        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        key_t actual;
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree->depth; ++current_depth) {
            const key_t* node_start = tree->keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

            lb = cooperative_linear_search(active, key, node_start, tree->ntg_degree[current_depth]);
            actual = node_start[lb];

            unsigned new_pos = tree->children[pos] + lb;
//            active = active && new_pos < tree->size; // TODO

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && pos < tree->size && key == actual) {
            return tree->values[pos];
        }

        return not_found;
    }
#endif

    __host__ unsigned count_ntg_steps(const key_t x, const key_t* arr, const unsigned ntg_degree) {
#if 0
        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        unsigned lower_bound = max_keys;
	const unsigned my_lane_id = lane_id();
	const unsigned ntg_size = 1u << ntg_degree;
        unsigned leader = ntg_size*(my_lane_id >> ntg_degree);
        const int lane_offset = my_lane_id - leader;
        const uint32_t window_mask = (ntg_size - 1u) << leader;
        assert(my_lane_id >= leader);

        // iterate over all threads within a cooperative group by shifting the leader thread from the lsb to the msb within the window mask
        for (unsigned shift = 0; shift < ntg_size; ++shift) {
            int key_idx = lane_offset - ntg_size;
            const key_t leader_x = __shfl_sync(window_mask, x, leader);
            const key_t* leader_arr = reinterpret_cast<const key_t*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));

            const auto leader_active = __shfl_sync(window_mask, active, leader);
            unsigned exhausted_cnt = leader_active ? 0 : ntg_size;
            uint32_t matches = 0;
            while (matches == 0 && exhausted_cnt < ntg_size) {
                key_idx += ntg_size;

                key_t value;
                if (key_idx < max_keys) value = leader_arr[key_idx];
                matches = __ballot_sync(window_mask, key_idx < max_keys && value >= leader_x);
                exhausted_cnt = __popc(__ballot_sync(window_mask, key_idx >= max_keys));
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
#endif

	const unsigned ntg_size = 1u << ntg_degree;
	const unsigned lb = std::lower_bound(arr, arr + max_keys, x) - arr;
//        printf("lb: %u\n", lb);
	return 1 + std::min(lb, ntg_size - 1)/ntg_size;
    }


//    __host__ unsigned optimize_ntg_degree_for_level(const unsigned target_depth, const unsigned current_ntg_degree) {
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
	return count_ntg_steps(key, node_start, current_ntg_degree);
    }

    __host__ void optimize_ntg(const std::vector<key_t>& sample) {
        std::vector<unsigned> ntg_degrees;
//	unsigned current_ntg_degree = 5;
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
            do {
                current_ntg_degree -= 1;
                for (const key_t key : sample) {
                    acc_steps += count_ntg_steps_at_target_depth(current_depth, current_ntg_degree, key);
                }
                avg_steps_after = static_cast<double>(acc_steps)/sample.size();
                std::cout << "depth " << current_depth << " avg_steps_after : " << avg_steps_after << std::endl;

            } while (2*avg_steps_before/avg_steps_after > 1. && current_ntg_degree > 0);
            current_ntg_degree += 1; // the final narrowing does not improve the throughput

	    std::cout << "depth " << current_depth << " final ntg_degree: " << current_ntg_degree << std::endl;
	    ntg_degrees.push_back(current_ntg_degree);
        }
    }
#endif
};

}

