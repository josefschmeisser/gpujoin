#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>

#include "cuda_utils.cuh"

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

namespace harmonia {

template<class Key, class Value, unsigned fanout>
struct harmonia_tree {
    using key_t = Key;
    using value_t = Value;
    using child_ref_t = uint32_t;

    static constexpr auto max_keys = fanout - 1;

    key_t* keys;
    child_ref_t* children;
    value_t* values;
    unsigned size;
    unsigned depth;

    struct device_handle {
        key_t* keys;
        child_ref_t* children;
        value_t* values;
        unsigned size;
        unsigned depth;
    };

    struct intermediate_node {
        bool is_leaf = false;
        unsigned count = 0;
        unsigned tree_size = 0;
        key_t keys[fanout - 1];
//        std::unique_ptr<itermediate_node> children[fanout];
        union {
            intermediate_node* children[fanout];
            value_t values[fanout];
        };
    };

    using tree_level_t = std::vector<std::unique_ptr<intermediate_node>>;
    using tree_levels_t = std::vector<std::unique_ptr<tree_level_t>>;

    ~harmonia_tree() {
    }

    key_t max_key(const intermediate_node& subtree) {
        return 0; // TODO
    }

    void add(intermediate_node& node, key_t key, value_t value) {
    }

    void add(intermediate_node& node, key_t key, intermediate_node* child) {
    }

    void construct_inner_nodes(tree_levels_t& tree_levels) {
        const auto& lower_level = tree_levels.front();

        if (lower_level->size() == 1) {
            return;
        }

        auto current_level = std::make_unique<tree_level_t>();
        auto node = std::make_unique<intermediate_node>();
        node->is_leaf = false;

        for (unsigned i = 0; i < lower_level->size() - 1; i++) {
            auto& curr = lower_level->at(i);
            if (node->count >= max_keys) {
                node->children[node->count] = curr.get();
                current_level->push_back(std::move(node));
                node = std::make_unique<intermediate_node>();
                node->is_leaf = false;
            } else {
                key_t sep = max_key(*curr);
                add(*node, sep, curr.get());
            }
        }
        current_level->push_back(std::move(node));

        tree_levels.push_back(std::move(current_level));

        construct_inner_nodes(tree_levels);
    }

    tree_levels_t construct_levels(const std::vector<key_t>& keys) {
        uint64_t n = keys.size();

        auto leaves = std::make_unique<tree_level_t>();
        auto node = std::make_unique<intermediate_node>();
        node->is_leaf = true;

        // construct leaves
        for (uint64_t i = 0; i < n; i++) {
            auto k = keys[i];
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

    void store_node(const intermediate_node& node, unsigned key_offset /*, unsigned& children_offset*/) {
        for (unsigned i = 0; i < node.count; ++i) {
            std::memcpy(keys + key_offset, node.keys, sizeof(key_t)*max_keys);
        }
        for (unsigned i = 0; i < node.count; ++i) {
            store_node(*node.children[i], key_offset += node.count);
        }/*
        children[children_offset] = key_offset;
        children_offset += 1;*/
    }

    /// offset into the flat key-array
    void store_nodes(const tree_level_t& tree_level, unsigned& key_offset, unsigned& children_offset) {
        for (auto& node : tree_level) {
            store_node(*node, key_offset);
            key_offset += max_keys;
        }

        if (tree_level.front()->is_leaf) {
            child_ref_t values_prefix_sum = 0;
            for (auto& node : tree_level) {
                children[children_offset++] = values_prefix_sum;
                values_prefix_sum += fanout;
            }
        } else {
            // write out prefix sum array entries
            auto prefix_sum = key_offset + 1;
            children[children_offset++] = prefix_sum;
            for (unsigned i = 1; i < tree_level.size(); ++i) {
                prefix_sum += max_keys;
                children[children_offset++] = prefix_sum;
            }
        }
    }

    void construct(const std::vector<key_t>& input) {
        auto tree_levels = construct_levels(input);
        auto& root = tree_levels.front()->front();

        // allocate arrays
        const auto key_array_size = sizeof(key_t)*max_keys*root->tree_size;
        keys = (key_t*)malloc(key_array_size);
        const auto node_count = 0;
        children = (child_ref_t*)malloc(node_count*sizeof(child_ref_t));
        values = (value_t*)malloc(input.size()*sizeof(value_t));

        unsigned key_offset, children_offset;
        for (auto& tree_level : tree_levels) {
            store_nodes(*tree_level, key_offset, children_offset);
        }

        for (unsigned i = 0; i < input.size(); ++i) {
            values[i] = i;
        }
        depth = tree_levels.size();
        // TODO
    }

    template<unsigned degree>
    __device__ value_t cooperative_linear_search(bool active, key_t x, const key_t* arr) {
        enum { WINDOW_SIZE = 1 << degree };

        assert(__all_sync(FULL_MASK, 1));

        const unsigned my_lane_id = lane_id();

        unsigned lower_bound = max_keys;

        unsigned leader = WINDOW_SIZE*(my_lane_id >> degree);
        const uint32_t window_mask = __funnelshift_l(FULL_MASK, 0, WINDOW_SIZE) << leader; // TODO replace __funnelshift_l() with compile time computation

        assert(my_lane_id >= leader);
        const int lane_offset = my_lane_id - leader;

        for (unsigned shift = 0; shift < WINDOW_SIZE; ++shift) {
            int key_idx = lane_offset - WINDOW_SIZE;
            const key_t leader_x = __shfl_sync(window_mask, x, leader);
            const key_t* leader_arr = reinterpret_cast<const key_t*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));
            const unsigned leader_size = __shfl_sync(window_mask, max_keys, leader);

            const auto leader_active = __shfl_sync(window_mask, active, leader);
            unsigned exhausted_cnt = leader_active ? 0 : WINDOW_SIZE;

            uint32_t matches = 0;
            while (matches == 0 && exhausted_cnt < WINDOW_SIZE) {
                key_idx += WINDOW_SIZE;

                key_t value;
                if (key_idx < leader_size) value = leader_arr[key_idx];
                matches = __ballot_sync(window_mask, key_idx < leader_size && value >= leader_x);
                exhausted_cnt = __popc(__ballot_sync(window_mask, key_idx >= leader_size));
            }

            if (my_lane_id == leader && matches != 0) {
                lower_bound = key_idx + __ffs(matches) - 1 - leader;
            }

            leader += 1;
        }

        return lower_bound;
    }

    template<unsigned degree = 3>
    __device__ value_t lookup(bool active, const device_handle tree, key_t key) {
   //     unsigned current_depth = 0;

        assert(__all_sync(FULL_MASK, 1)); // ensure that all threads participate

        unsigned lb, pos = 0;
        for (unsigned current_depth = 0; current_depth <= tree.depth; ++current_depth) {
            lb = cooperative_linear_search(active, key, tree.keys + pos);
            unsigned new_pos = tree.children[pos + lb];
            active = active && new_pos < tree.size;

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && (pos < tree.size) && (tree.keys[pos] == key)) {
            return 0; // TODO node->payloads[pos];
        }

        return -1; // TODO
    }

    __host__ value_t lookup(key_t key) {
        bool active = true;
        unsigned lb, pos = 0;
        for (unsigned current_depth = 0; current_depth <= depth; ++current_depth) {
            lb = cooperative_linear_search(active, key, keys + pos);
            unsigned new_pos = children[pos + lb];
            active = active && new_pos < size;

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            pos = active ? new_pos : 0;
        }

        if (active && (pos < size) && (keys[pos] == key)) {
            return 0; // TODO return node->payloads[pos];
        }

        return -1; // TODO
    }
};

} // end namespace harmonia

