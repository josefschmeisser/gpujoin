#pragma once

#include <memory>

namespace harmonia {

template<class Key, class Value, unsigned fanout>
struct harmonia_tree {
    using key_t = Key;
    using value_t = Value;

    static constexpr auto max_keys = fanout -1;

    Key* keys;
    Value* values;
    unsigned size;

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

    key_t max_key(const intermediate_node* subtree);

    void construct_inner_nodes(tree_levels_t& tree_levels) {
        const auto& lower_level = tree_levels.front();

        if (lower_level.size() == 1) {
            return lower_level.front();
        }

        tree_level_t current_level;
        auto node = std::make_unique<intermediate_node>();
        node->is_leaf = false;

        for (unsigned i = 0; i < lower_level.size() - 1; i++) {
            auto curr = lower_level[i];
            bool full = node->count >= Node::maxEntries;
            if (full) {
                node->children[node->count] = curr.get();
                current_level.push_back(std::move(node));
                node = std::make_unique<intermediate_node>();
                node->is_leaf = false;
            } else {
                key_t sep = max_key(curr);
                append_into(node, sep, curr);
            }
        }
        node->payloads[node->header.count] = reinterpret_cast<payload_t>(lower_level[lower_level.size() - 1]);
        current_level.push_back(node);

        tree_levels.push_back(std::move(current_level));

        construct_inner_nodes(tree_levels);
    }

    tree_levels_t construct_levels(const vector<key_t>& keys) {
        uint64_t n = keys.size();

        tree_level_t leaves;
        auto node = std::make_unique<intermediate_node>();
        node->is_leaf = true;

        // construct leaves
        for (uint64_t i = 0; i < n; i++) {
            auto k = keys[i];
            value_t value = i;
            bool full = node->count >= max_keys;
            if (full) {
                leaves.push_back(std::move(node));
                node = std::make_unique<intermediate_node>();
                node->is_leaf = true;
            }
            append_into(node, k, value);
        }
        leaves.push_back(std::move(node));

        tree_levels_t tree_levels { std::move(leaves) };

        // recursively construct the remaining levels in bottom-up fashion
        construct_inner_nodes(tree_levels);
//        tree_levels.push_back(std::move(root_level));

        return tree_levels;
    }

    void store_node(const intermediate_node& node, unsigned key_array_pos) {
        for (unsigned i = 0; i < node.count; ++i) {
            std::memcpy(keys + current_key_offset, node.keys, sizeof(key_t)*max_keys);
        }
        for (unsigned i = 0; i < node.count; ++i) {
            store_node(*node.children[i], key_array_pos += node.count);
        }
    }

    void store_nodes(const intermediate_node& root) {
        const auto key_array_size = sizeof(key_t)*max_keys*root->tree_size;
        keys = malloc(key_array_size);
        store_node(root, 0);
    }

    template<Key, Value>
    __device__ payload_t lookup(const device_handle tree, Key key) {

    }

} // end namespace harmonia

