#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>

#include "cuda_utils.cuh"
#include "search.cuh"

namespace index_structures {

template<
    class Key,
    class Value,
    Value Not_Found>
struct btree {
    using key_t = Key;
    using value_t = Value;

    struct NodeBase {
        union {
            struct {
                uint16_t count;
                bool isLeaf;
            } header;
            uint8_t header_with_padding[GPU_CACHE_LINE_SIZE];
        };
    };
    static_assert(sizeof(NodeBase) == GPU_CACHE_LINE_SIZE);

    struct LeafNode : public NodeBase {
        static const unsigned pageSize = 4 * 1024;
        static const unsigned maxEntries = ((pageSize - sizeof(NodeBase)) / (sizeof(key_t) + sizeof(value_t))) - 1;

        key_t keys[maxEntries];
        value_t payloads[maxEntries];
    };
    static_assert(sizeof(LeafNode) < LeafNode::pageSize);

    struct InnerNode : public NodeBase {
        static const unsigned pageSize = 4 * 1024;
        static const unsigned maxEntries = ((pageSize - sizeof(NodeBase) - sizeof(NodeBase*)) / (sizeof(key_t) + sizeof(NodeBase*))) - 1;

        key_t keys[maxEntries];
        NodeBase* children[maxEntries + 1];
    };
    static_assert(sizeof(InnerNode) < InnerNode::pageSize);

    NodeBase* root = nullptr;

    ~btree() {
        free_tree(root);
    }

    __host__ void free_tree(NodeBase* node) {
        // TODO
    }

    LeafNode* create_leaf() {
        LeafNode* node;
        void** dst = reinterpret_cast<void**>(&node);

        // TODO use allocator
        cudaMallocManaged(dst, LeafNode::pageSize);
        //node = reinterpret_cast<Node*>(numa_alloc_onnode(Node::pageSize, 0));
        node->header.isLeaf = true;

        // validate alignment
        if ((reinterpret_cast<uintptr_t>(node) & GPU_CACHE_LINE_SIZE-1) != 0) { throw std::runtime_error("unaligned memory"); }
        if ((reinterpret_cast<uintptr_t>(&node->keys[0]) & GPU_CACHE_LINE_SIZE-1) != 0) { throw std::runtime_error("unaligned memory"); }

        // initialize key vector with the largest key value possible
        static constexpr auto maxKey = std::numeric_limits<key_t>::max();
        for (unsigned i = 0; i < LeafNode::maxEntries; ++i) {
            node->keys[i] = maxKey;
        }

        return node;
    }

    InnerNode* create_inner() {
        InnerNode* node;
        void** dst = reinterpret_cast<void**>(&node);

        // TODO use allocator
        cudaMallocManaged(dst, InnerNode::pageSize);
        //node = reinterpret_cast<Node*>(numa_alloc_onnode(Node::pageSize, 0));
        node->header.isLeaf = false;

        // validate alignment
        if ((reinterpret_cast<uintptr_t>(node) & GPU_CACHE_LINE_SIZE-1) != 0) { throw std::runtime_error("unaligned memory"); }
        if ((reinterpret_cast<uintptr_t>(&node->keys[0]) & GPU_CACHE_LINE_SIZE-1) != 0) { throw std::runtime_error("unaligned memory"); }

        // initialize key vector with the largest key value possible
        static constexpr auto maxKey = std::numeric_limits<key_t>::max();
        for (unsigned i = 0; i < LeafNode::maxEntries; ++i) {
            node->keys[i] = maxKey;
        }

        return node;
    }

    bool append_into(LeafNode* dst, key_t key, value_t value) {
        assert(dst->header.isLeaf);
        if (dst->header.count >= LeafNode::maxEntries) { return false; }

        dst->keys[dst->header.count] = key;
        dst->payloads[dst->header.count] = value;
        dst->header.count += 1;
        return true;
    }

    bool append_into(InnerNode* dst, key_t key, NodeBase* child) {
        assert(!dst->header.isLeaf);
        if (dst->header.count >= LeafNode::maxEntries) { return false; }

        dst->keys[dst->header.count] = key;
        dst->children[dst->header.count] = child;
        dst->header.count += 1;
        return true;
    }

    static key_t max_key(const NodeBase* tree) {
        if (tree->header.isLeaf) {
            const LeafNode* leaf = static_cast<const LeafNode*>(tree);
            return leaf->keys[tree->header.count - 1];
        }
        return max_key(static_cast<const InnerNode*>(tree)->children[tree->header.count]);
    }

    NodeBase* construct_inner_nodes(std::vector<NodeBase*> lower_level, float load_factor) {
        if (lower_level.size() == 1) {
            return lower_level.front();
        }

        std::vector<NodeBase*> current_level;
        InnerNode* node = create_inner();
        for (unsigned i = 0; i < lower_level.size() - 1; i++) {
            NodeBase* curr = lower_level[i];
            key_t sep = max_key(curr);
            bool full = node->header.count >= InnerNode::maxEntries;
            full = full || static_cast<float>(node->header.count) / static_cast<float>(InnerNode::maxEntries) > load_factor;
            if (full) {
                node->children[node->header.count] = curr;
                current_level.push_back(node);
                node = create_inner();
            } else {
                bool appended = append_into(node, sep, curr);
                (void)appended;
                assert(appended);
            }
        }
        node->children[node->header.count] = lower_level[lower_level.size() - 1];
        current_level.push_back(node);
        std::cout << "count per inner node: " << lower_level.size() / current_level.size() << std::endl;

        return construct_inner_nodes(current_level, load_factor);
    }

    void construct(const std::vector<key_t>& keys, float load_factor) {
        assert(load_factor > 0 && load_factor <= 1.0);
        uint64_t n = keys.size();

        std::vector<NodeBase*> leaves;
        LeafNode* node = create_leaf();
        for (uint64_t i = 0; i < n; i++) {
            auto k = keys[i];
            value_t value = i;
            bool full = node->header.count >= LeafNode::maxEntries;
            full = full || static_cast<float>(node->header.count) / static_cast<float>(LeafNode::maxEntries) > load_factor;
            if (full) {
                leaves.push_back(node);
                node = create_leaf();
                bool inserted = append_into(node, k, value);
                (void)inserted;
                assert(inserted);
            } else {
                bool appended = append_into(node, k, value);
                (void)appended;
                assert(appended);
            }
        }
        leaves.push_back(node);

        std::cout << "count per leaf node: " << n / leaves.size() << std::endl;

        if (root) {
            free_tree(root);
        }
        root = construct_inner_nodes(leaves, load_factor);

        std::cout << "tree size: " << tree_size_in_byte(root) / (1024*1024) << " MB" << std::endl;
    }

    void construct_dense(uint32_t numElements, float load_factor) {
        std::vector<uint32_t> keys(numElements);
        std::iota(keys.begin(), keys.end(), 0);
        construct(keys, load_factor);
    }

/*
    static unsigned lower_bound(Node* node, key_t key) {
        //cout << "search key: " << key << " in [" << node->keys[0] << ", " << node->keys[node->header.count - 1] << "]" << endl;
        unsigned lower = 0;
        unsigned upper = node->header.count;
        do {
            unsigned mid = ((upper - lower) / 2) + lower;
            if (key < node->keys[mid]) {
                upper = mid;
            } else if (key > node->keys[mid]) {
                lower = mid + 1;
            } else {
                return mid;
            }
        } while (lower < upper);
        return lower;
    }
*/

    __host__ bool lookup(key_t key, value_t& result) {
        assert(root);
        NodeBase* node = root;
        while (!node->header.isLeaf) {
            const InnerNode* inner = static_cast<const InnerNode*>(node);
            unsigned pos = std::lower_bound(inner->keys, inner->keys + inner->header.count, key) - inner->keys;
            //cout << "inner pos: " << pos << endl;
            node = inner->children[pos];
            if (node == nullptr) {
                return false;
            }
        }

        const LeafNode* leaf = static_cast<const LeafNode*>(node);
        unsigned pos = std::lower_bound(leaf->keys, leaf->keys + leaf->header.count, key) - leaf->keys;
        //cout << "pos: " << pos << endl;
        if ((pos < leaf->header.count) && (leaf->keys[pos] == key)) {
            result = leaf->payloads[pos];
            return true;
        }

        return false;
    }

#if 0
    // can't be compiled by nvcc...
    void prefetch_tree(Node* tree, int device) {
        if (device < 0) {
            cudaGetDevice(&device);
        }

        const auto prefetchNode = [&](const auto& self, Node* node) -> void {
            cudaMemAdvise(node, Node::pageSize, cudaMemAdviseSetReadMostly, device);
            cudaMemPrefetchAsync(node, btree::Node::pageSize, device);

            if (node->header.isLeaf) return;
            for (unsigned i = 0; i <= node->header.count; ++i) {
                Node* child = reinterpret_cast<Node*>(node->payloads[i]);
                assert(child);
                self(self, child);
            }
        };
        prefetchNode(prefetchNode, tree);
    }
#endif

    void prefetch_subtree(NodeBase* node, int device) {
        bool isLeaf = node->header.isLeaf;

        if (!isLeaf) {
            const InnerNode* inner = static_cast<const InnerNode*>(node);
            for (unsigned i = 0; i <= inner->header.count; ++i) {
                NodeBase* child = inner->payloads[i];
                assert(child);
                prefetchSubtree(child, device);
            }
        }

        cudaMemPrefetchAsync(node, isLeaf ? LeafNode::pageSize : InnerNode::pageSize, device);

        if (isLeaf) return;
    };

    void prefetch_tree(NodeBase* tree, int device = -1) {
        printf("prefetching btree nodes...\n");
        if (device < 0) {
            cudaGetDevice(&device);
        }
        prefetchSubtree(tree, device);
        cudaDeviceSynchronize();
    }

    NodeBase* copy_btree_to_gpu(NodeBase* tree) {
        NodeBase* newTree;
        cudaMalloc(&newTree, tree->header.isLeaf ? LeafNode::pageSize : InnerNode::pageSize);
        if (!tree->header.isLeaf) {
            const InnerNode* inner = static_cast<const InnerNode*>(tree);
            std::unique_ptr<uint8_t[]> tmpMem { new uint8_t[InnerNode::pageSize] };
            InnerNode* tmp = reinterpret_cast<InnerNode*>(tmpMem.get());
            std::memcpy(tmp, tree, InnerNode::pageSize);
            for (unsigned i = 0; i <= tree->header.count; ++i) {
                NodeBase* child = inner->children[i];
                NodeBase* newChild = copy_btree_to_gpu(child);
                tmp->children[i] = newChild;
            }
            cudaMemcpy(newTree, tmp, InnerNode::pageSize, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(newTree, tree, LeafNode::pageSize, cudaMemcpyHostToDevice);
        }
        return newTree;
    }

    size_t tree_size_in_byte(const NodeBase* tree) {
        if (tree->header.isLeaf) { return LeafNode::pageSize; }

        size_t size = InnerNode::pageSize;
        for (unsigned i = 0; i <= tree->header.count; ++i) {
            const InnerNode* inner = static_cast<const InnerNode*>(tree);
            const NodeBase* child = inner->children[i];
            assert(child);
            size += tree_size_in_byte(child);
        }
        return size;
    }

    static __device__ value_t lookup(const NodeBase* tree, key_t key) {
        //printf("lookup key: %lu\n", key);
        const NodeBase* node = tree;
        while (!node->header.isLeaf) {
            const InnerNode* inner = static_cast<const InnerNode*>(node);
            unsigned pos = branchy_binary_search(key, inner->keys, inner->header.count);
            //unsigned pos = linear_search(key, inner->keys, inner->header.count);
            //printf("inner pos: %d\n", pos);
            node = inner->children[pos];
            /*
            if (node == nullptr) {
                return Not_Found;
            }*/
        }

        const LeafNode* leaf = static_cast<const LeafNode*>(node);
        unsigned pos = branchy_binary_search(key, leaf->keys, leaf->header.count);
        //unsigned pos = linear_search(key, leaf->keys, leaf->header.count);
        //printf("leaf pos: %d\n", pos);
        if ((pos < leaf->header.count) && (leaf->keys[pos] == key)) {
            return leaf->payloads[pos];
        }

        return Not_Found;
    }

    static __device__ value_t lookup_with_hints(const NodeBase* tree, key_t key) {
        //printf("lookup_with_hints key: %lu\n", key);
        float hint = 0.5f;
        const NodeBase* node = tree;
        while (!node->header.isLeaf) {
            const InnerNode* inner = static_cast<const InnerNode*>(node);
            unsigned pos = branch_free_exponential_search(key, inner->keys, inner->header.count, hint);
            if (pos > 0 && pos < inner->header.count) {
                const auto prev = static_cast<float>(inner->keys[pos - 1]);
                const auto current = static_cast<float>(inner->keys[pos]);
                hint = (static_cast<float>(key) - prev)/(current - prev);
                //printf("prev: %f current: %f hint: %f\n", prev, current, hint);
            } else {
                hint = 0.5f;
            }

            node = node->children[pos];
            /*
            if (node == nullptr) {
                return Not_Found;
            }*/
        }

        const LeafNode* leaf = static_cast<const LeafNode*>(node);
        //unsigned pos = naive_lower_bound(node, key);
        unsigned pos = branch_free_exponential_search(key, leaf->keys, leaf->header.count, hint);
        //printf("leaf pos: %d\n", pos);
    /*
        if ((pos < leaf->header.count) && (leaf->keys[pos] == key)) {
            return leaf->payloads[pos];
        }
        return Not_Found;
    */
        return (pos < leaf->header.count) && (leaf->keys[pos] == key) ? leaf->payloads[pos] : Not_Found;
    }

    // this function has to be called by the entire warp, otherwise the function is likly to yield wrong results
    static __device__ value_t cooperative_lookup(bool active, const NodeBase* tree, key_t key) {
        assert(__all_sync(FULL_MASK, 1));

        //printf("btree_cooperative_lookup active: %d key: %u\n", active, key);
        const NodeBase* node = tree;
        while (__any_sync(FULL_MASK, active && !node->header.isLeaf)) {
            const InnerNode* inner = static_cast<const InnerNode*>(node);
            unsigned pos = cooperative_linear_search(active, key, inner->keys, inner->header.count);
            //printf("inner pos: %d\n", pos);

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            node = active ? inner->children[pos] : tree;
        }

        const LeafNode* leaf = static_cast<const LeafNode*>(node);
        unsigned pos = cooperative_linear_search(active, key, leaf->keys, leaf->header.count);
        //printf("leaf pos: %d\n", pos);
        if (active && (pos < leaf->header.count) && (leaf->keys[pos] == key)) {
            return leaf->payloads[pos];
        }

        return Not_Found;
    }

#if 0
    __device__ value_t btree_lookup_with_page_replication(const Node* tree, key_t key) {
        __shared__ uint8_t page_cache[32][Node::pageSize];

        const Node* node = tree;
        while (!node->header.isLeaf) {
            unsigned pos = branchy_binary_search(key, node->keys, node->header.count);
            //unsigned pos = linear_search(key, node->keys, node->header.count);
            node = reinterpret_cast<const Node*>(node->payloads[pos]);
        }

        unsigned pos = branchy_binary_search(key, node->keys, node->header.count);
        //unsigned pos = linear_search(key, node->keys, node->header.count);
        if ((pos < node->header.count) && (node->keys[pos] == key)) {
            return node->payloads[pos];
        }

        return Not_Found;
    }
#endif

#if 0
    __device__ value_t btree_lookup_with_hints(const Node* tree, key_t key) {
        unsigned pos = branch_free_binary_search(key, tree->keys, tree->header.count);

        auto prev = (pos > 0) ? tree->keys[pos - 1] : key>>1;
        auto current = (pos < tree->header.count) ? tree->keys[pos] : key + 1;
        //auto current = tree->keys[pos];
        float hint = (static_cast<float>(key) - prev)/(current - prev);

        const Node* node = reinterpret_cast<const Node*>(tree->payloads[pos]);
        while (!node->header.isLeaf) {
            pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);

            auto prev = (pos > 0) ? node->keys[pos - 1] : key>>1;
            auto current = (pos < node->header.count) ? node->keys[pos] : key + 1;
            //auto current = node->keys[pos];
            hint = (static_cast<float>(key) - prev)/(current - prev);

            node = reinterpret_cast<const Node*>(node->payloads[pos]);
        }

        pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);
        return (pos < node->header.count) && (node->keys[pos] == key) ? node->payloads[pos] : Not_Found;
    }
#endif

#if 0
    __device__ value_t btree_lookup_with_hints(const Node* tree, key_t key) {
        //printf("btree_lookup key: %lu\n", key);
        float hint = 0.5f;
        const Node* node = tree;
        while (!node->header.isLeaf) {
            unsigned pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);

            const auto prev = (pos > 0) ? node->keys[pos - 1] : key>>1;
            const auto current = (pos < node->header.count) ? node->keys[pos] : key + 1;
            hint = (static_cast<float>(key) - prev)/(current - prev);

            node = reinterpret_cast<const Node*>(node->payloads[pos]);
        }

        //unsigned pos = naive_lower_bound(node, key);
        unsigned pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);

        return (pos < node->header.count) && (node->keys[pos] == key) ? node->payloads[pos] : Not_Found;
    }
#endif

#if 0
    static __device__ value_t btree_lookup_with_hints(const Node* tree, key_t key) {
        //printf("btree_lookup key: %lu\n", key);
        float hint = 0.5;
        const Node* node;
        unsigned pos;
        value_t payload = reinterpret_cast<value_t>(tree);
        do {
            node = reinterpret_cast<const Node*>(payload);

            pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);
            if (pos > 0 && pos < node->header.count) {
                const auto prev = static_cast<float>(node->keys[pos - 1]);
                const auto current = static_cast<float>(node->keys[pos]);
                hint = (static_cast<float>(key) - prev)/(current - prev);
            } else {
                hint = 0.5f;
            }

            payload = node->payloads[pos];
        } while (!node->header.isLeaf);

        if ((pos < node->header.count) && (node->keys[pos] == key)) {
            return payload;
        }

        return Not_Found;
    }
#endif

};

} // namespace index
