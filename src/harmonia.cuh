#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>
#include <numeric>
#include <type_traits>

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

    static constexpr auto max_keys = fanout - 1;

    std::vector<key_t> keys;
    std::vector<child_ref_t> children;
    std::vector<value_t> values;
    unsigned size;
    unsigned depth;

    struct device_handle_t {
        key_t* keys;
        child_ref_t* children;
        value_t* values;
        unsigned size;
        unsigned depth;
    }* device_handle = nullptr;

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
    }

    void create_device_handle() {//device_handle_t& handle) {
        // initialize fields
        device_handle_t tmp;
        tmp.depth = depth;
        tmp.size = size;
        auto ret = cudaMalloc(&tmp.keys, sizeof(key_t)*keys.size());
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(tmp.keys, keys.data(), sizeof(key_t)*keys.size(), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&tmp.children, sizeof(child_ref_t)*children.size());
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(tmp.children, children.data(), sizeof(key_t)*children.size(), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);

        if constexpr (!Sorted_Only) {
            ret = cudaMalloc(&tmp.values, sizeof(value_t)*values.size());
            assert(ret == cudaSuccess);
            ret = cudaMemcpy(tmp.values, values.data(), sizeof(key_t)*values.size(), cudaMemcpyHostToDevice);
            assert(ret == cudaSuccess);
        }

        // create cuda struct
        ret = cudaMalloc(&device_handle, sizeof(device_handle_t));
        assert(ret == cudaSuccess);
        ret = cudaMemcpy(device_handle, &tmp, sizeof(device_handle_t), cudaMemcpyHostToDevice);
        assert(ret == cudaSuccess);
    }

    template<unsigned Degree>
    __device__ static value_t cooperative_linear_search(bool active, key_t x, const key_t* arr) {
        enum { WINDOW_SIZE = 1 << Degree };

        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads within the warp participate

        const unsigned my_lane_id = lane_id();
        unsigned leader = WINDOW_SIZE*(my_lane_id >> Degree);
        const uint32_t window_mask = ((1u << WINDOW_SIZE) - 1u) << leader;
        assert(my_lane_id >= leader);
        const int lane_offset = my_lane_id - leader;
        unsigned lower_bound = max_keys;

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
    __device__ static std::enable_if_t<Sorted_Only && Degree < 6, value_t> lookup(bool active, const device_handle_t* tree, key_t key) {
        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        key_t actual;
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree->depth; ++current_depth) {
            key_t* node_start = tree->keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

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
        printf("todo\n");

        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        key_t actual;
        unsigned lb = 0, pos = 0;
        for (unsigned current_depth = 0; current_depth < tree->depth; ++current_depth) {
            key_t* node_start = tree->keys + max_keys*pos; // TODO use shift when max_keys is a power of 2

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
            using key_array_t = key_t[max_keys];
            key_t* raw = &keys[max_keys*pos];
            key_array_t& current_node = reinterpret_cast<key_array_t&>(*raw);

            lb = std::lower_bound(std::cbegin(current_node), std::cend(current_node), key) - std::cbegin(current_node);
            actual = current_node[lb];

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
};

} // end namespace harmonia

