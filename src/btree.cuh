#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>

#include "cuda_utils.cuh"

namespace index_structures {

template<
    class Key,
    class Value,
    unsigned Fanout,
    Value Not_Found>
struct btree {
    using key_t = Key;
    using value_t = Value;

    static constexpr auto max_keys = Fanout - 1;

    struct NodeBase {
        union {
            struct {
                uint16_t count;
                bool isLeaf;
            } header;
            uint8_t header_with_padding[128];
        };
    };
    static_assert(sizeof(NodeBase) == 128);

    struct Node : public NodeBase {
        static const uint64_t pageSize = 4 * 1024;
        static const uint64_t maxEntries = ((pageSize - sizeof(NodeBase) - sizeof(value_t)) / (sizeof(key_t) + sizeof(value_t))) - 1;

        key_t keys[maxEntries];
        value_t payloads[maxEntries + 1];
    };
    static_assert(sizeof(Node) < Node::pageSize);

    Node* root = nullptr;

    ~btree() {
        // TODO
    }

    Node* create_node(bool isLeaf) {
        Node* node;
        void** dst = reinterpret_cast<void**>(&node);
        cudaMallocManaged(dst, Node::pageSize);
        //node = reinterpret_cast<Node*>(numa_alloc_onnode(Node::pageSize, 0));
        node->header.isLeaf = isLeaf;

        // validate alignment
        if ((reinterpret_cast<uintptr_t>(node) & GPU_CACHE_LINE_SIZE-1) != 0) { throw std::runtime_error("unaligned memory"); }
        if ((reinterpret_cast<uintptr_t>(&node->keys[0]) & GPU_CACHE_LINE_SIZE-1) != 0) { throw std::runtime_error("unaligned memory"); }

        // initialize key vector with the largest key value possible
        static constexpr auto maxKey = std::numeric_limits<key_t>::max();
        for (unsigned i = 0; i < Node::maxEntries; ++i) {
            node->keys[i] = maxKey;
        }

        return node;
    }

    bool append_into(Node* dst, key_t key, value_t value) {
        if (dst->header.count >= Node::maxEntries) { return false; }

        dst->keys[dst->header.count] = key;
        dst->payloads[dst->header.count] = value;
        dst->header.count += 1;
        return true;
    }

    static key_t max_key(Node* tree) {
        if (tree->header.isLeaf) {
            return tree->keys[tree->header.count - 1];
        }
        return max_key(reinterpret_cast<Node*>(tree->payloads[tree->header.count]));
    }

    Node* construct_inner_nodes(std::vector<Node*> lower_level, float load_factor) {
        if (lower_level.size() == 1) {
            return lower_level.front();
        }

        std::vector<Node*> current_level;
        Node* node = create_node(false);
        for (unsigned i = 0; i < lower_level.size() - 1; i++) {
            Node* curr = lower_level[i];
            key_t sep = max_key(curr);
            bool full = node->header.count >= Node::maxEntries;
            full = full || static_cast<float>(node->header.count) / static_cast<float>(Node::maxEntries) > load_factor;
            if (full) {
                //node->upperOrNext = curr;
                node->payloads[node->header.count] = reinterpret_cast<value_t>(curr);
                current_level.push_back(node);
                node = create_node(false);
            } else {
                bool appended = append_into(node, sep, reinterpret_cast<value_t>(curr));
                (void)appended;
                assert(appended);
            }
        }
        //node->upperOrNext = lower_level[lower_level.size() - 1];
        node->payloads[node->header.count] = reinterpret_cast<value_t>(lower_level[lower_level.size() - 1]);
        current_level.push_back(node);
        std::cout << "count per inner node: " << lower_level.size() / current_level.size() << std::endl;

        return construct_inner_nodes(current_level, load_factor);
    }

    Node* construct(const std::vector<key_t>& keys, float load_factor) {
        assert(load_factor > 0 && load_factor <= 1.0);
        uint64_t n = keys.size();

        std::vector<Node*> leaves;
        Node* node = create_node(true);
        for (uint64_t i = 0; i < n; i++) {
            auto k = keys[i];
            value_t value = i;
            bool full = node->header.count >= Node::maxEntries;
            full = full || static_cast<float>(node->header.count) / static_cast<float>(Node::maxEntries) > load_factor;
            if (full) {
                leaves.push_back(node);
                node = create_node(true);
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

        Node* root = construct_inner_nodes(leaves, load_factor);

        std::cout << "tree size: " << tree_size_in_byte(root) / (1024*1024) << " MB" << std::endl;
        return root;
    }

    Node* construct_dense(uint32_t numElements, float load_factor) {
        std::vector<uint32_t> keys(numElements);
        std::iota(keys.begin(), keys.end(), 0);
        return construct(keys, load_factor);
    }

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

    __host__ bool lookup(Node* tree, key_t key, value_t& result) {
        Node* node = tree;
        while (!node->header.isLeaf) {
            unsigned pos = lower_bound(node, key);
            //cout << "inner pos: " << pos << endl;
            node = reinterpret_cast<Node*>(node->payloads[pos]);
            if (node == nullptr) {
                return false;
            }
        }

        unsigned pos = lower_bound(node, key);
        //cout << "pos: " << pos << endl;
        if ((pos < node->header.count) && (node->keys[pos] == key)) {
            result = node->payloads[pos];
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

    void prefetch_subtree(Node* node, int device) {
        bool isLeaf = node->header.isLeaf;

        if (!isLeaf) {
            for (unsigned i = 0; i <= node->header.count; ++i) {
                Node* child = reinterpret_cast<Node*>(node->payloads[i]);
                assert(child);
                prefetchSubtree(child, device);
            }
        }

        cudaMemPrefetchAsync(node, btree::Node::pageSize, device);

        if (isLeaf) return;
    };

    void prefetch_tree(Node* tree, int device = -1) {
        printf("prefetching btree nodes...\n");
        if (device < 0) {
            cudaGetDevice(&device);
        }
        prefetchSubtree(tree, device);
        cudaDeviceSynchronize();
    }

    Node* copy_btree_to_gpu(Node* tree) {
        Node* newTree;
        cudaMalloc(&newTree, Node::pageSize);
        if (!tree->header.isLeaf) {
            std::unique_ptr<uint8_t[]> tmpMem { new uint8_t[Node::pageSize] };
            Node* tmp = reinterpret_cast<Node*>(tmpMem.get());
            std::memcpy(tmp, tree, Node::pageSize);
            for (unsigned i = 0; i <= tree->header.count; ++i) {
                Node* child = reinterpret_cast<Node*>(tree->payloads[i]);
                Node* newChild = copy_btree_to_gpu(child);
                tmp->payloads[i] = reinterpret_cast<decltype(tmp->payloads[0])>(newChild);
            }
            cudaMemcpy(newTree, tmp, Node::pageSize, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(newTree, tree, Node::pageSize, cudaMemcpyHostToDevice);
        }
        return newTree;
    }

    size_t tree_size_in_byte(Node* tree) {
        if (tree->header.isLeaf) { return Node::pageSize; }

        size_t size = Node::pageSize;
        for (unsigned i = 0; i <= tree->header.count; ++i) {
            Node* child = reinterpret_cast<Node*>(tree->payloads[i]);
            assert(child);
            size += tree_size_in_byte(child);
        }
        return size;
    }

    __device__ value_t btree_lookup(const Node* tree, key_t key) {
        //printf("btree_lookup key: %lu\n", key);
        const Node* node = tree;
        while (!node->header.isLeaf) {
            unsigned pos = branchy_binary_search(key, node->keys, node->header.count);
            //unsigned pos = linear_search(key, node->keys, node->header.count);
            //printf("inner pos: %d\n", pos);
            node = reinterpret_cast<const Node*>(node->payloads[pos]);/*
            if (node == nullptr) {
                return Not_Found;
            }*/
        }

        unsigned pos = branchy_binary_search(key, node->keys, node->header.count);
        //unsigned pos = linear_search(key, node->keys, node->header.count);
        //printf("leaf pos: %d\n", pos);
        if ((pos < node->header.count) && (node->keys[pos] == key)) {
            return node->payloads[pos];
        }

        return Not_Found;
    }

    __device__ value_t btree_lookup_with_hints(const Node* tree, key_t key) {
        //printf("btree_lookup key: %lu\n", key);
        float hint = 0.5f;
        const Node* node = tree;
        while (!node->header.isLeaf) {
            unsigned pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);
            if (pos > 0 && pos < node->header.count) {
                const auto prev = static_cast<float>(node->keys[pos - 1]);
                const auto current = static_cast<float>(node->keys[pos]);
                hint = (static_cast<float>(key) - prev)/(current - prev);
    //            printf("prev: %f current: %f hint: %f\n", prev, current, hint);
            } else {
                hint = 0.5f;
            }

            node = reinterpret_cast<const Node*>(node->payloads[pos]);
            /*
            if (node == nullptr) {
                return Not_Found;
            }*/
        }

        //unsigned pos = naive_lower_bound(node, key);
        unsigned pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);
        //printf("leaf pos: %d\n", pos);
    /*
        if ((pos < node->header.count) && (node->keys[pos] == key)) {
            return node->payloads[pos];
        }
        return Not_Found;
    */
        return (pos < node->header.count) && (node->keys[pos] == key) ? node->payloads[pos] : Not_Found;
    }

    // this function has to be called by the entire warp, otherwise the function is likly to yield wrong results
    __device__ value_t btree_cooperative_lookup(bool active, const Node* tree, key_t key) {
        assert(__all_sync(FULL_MASK, 1));

    //    printf("btree_cooperative_lookup active: %d key: %u\n", active, key);
        const Node* node = tree;
        while (__any_sync(FULL_MASK, active && !node->header.isLeaf)) {
            unsigned pos = cooperative_linear_search(active, key, node->keys, node->header.count);
    //        printf("inner pos: %d\n", pos);

            // Inactive threads never progress during the traversal phase.
            // They, however, will be utilized by active threads during the cooperative search.
            node = active ? reinterpret_cast<const Node*>(node->payloads[pos]) : tree;
        }

        unsigned pos = cooperative_linear_search(active, key, node->keys, node->header.count);
    //    printf("leaf pos: %d\n", pos);
        if (active && (pos < node->header.count) && (node->keys[pos] == key)) {
            return node->payloads[pos];
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
    __device__ value_t btree_lookup_with_hints(const Node* tree, key_t key) {
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
