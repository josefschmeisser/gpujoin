#include "btree.cuh"

#include <cstdio>
#include <vector>
#include <iostream>
#include <tuple>
#include <cassert>
#include <numeric>

#include <numa.h>

using namespace std;

namespace btree {

Node* create_node(bool isLeaf) {
    Node* node;
    void** dst = reinterpret_cast<void**>(&node);
    cudaMallocManaged(dst, Node::pageSize);
    //node = reinterpret_cast<Node*>(numa_alloc_onnode(Node::pageSize, 0));
    node->isLeaf = isLeaf;
    return node;
}

bool append_into(Node* dst, key_t key, payload_t value) {
    if (dst->count >= Node::maxEntries) { return false; }

    dst->keys[dst->count] = key;
    dst->payloads[dst->count] = value;
    dst->count += 1;
    return true;
}

static key_t max_key(Node* tree) {
    if (tree->isLeaf) {
        return tree->keys[tree->count - 1];
    }
    return max_key(reinterpret_cast<Node*>(tree->payloads[tree->count]));
}

Node* construct_inner_nodes(vector<Node*> lowerLevel, float loadFactor) {
    if (lowerLevel.size() == 1) {
        return lowerLevel.front();
    }

    vector<Node*> currentLevel;
    Node* node = create_node(false);
    for (unsigned i = 0; i < lowerLevel.size() - 1; i++) {
        Node* curr = lowerLevel[i];
        key_t sep = max_key(curr);
        bool full = node->count >= Node::maxEntries;
        full = full || static_cast<float>(node->count) / static_cast<float>(Node::maxEntries) > loadFactor;
        if (full) {
            //node->upperOrNext = curr;
            node->payloads[node->count] = curr;
            currentLevel.push_back(node);
            node = create_node(false);
        } else {
            bool appended = append_into(node, sep, curr);
            (void)appended;
            assert(appended);
        }
    }
    //node->upperOrNext = lowerLevel[lowerLevel.size() - 1];
    node->payloads[node->count] = lowerLevel[lowerLevel.size() - 1];
    currentLevel.push_back(node);
    cout << "count per inner node: " << lowerLevel.size() / currentLevel.size() << endl;

    return construct_inner_nodes(currentLevel, loadFactor);
}

Node* construct(const vector<key_t>& keys, float loadFactor) {
    assert(loadFactor > 0 && loadFactor <= 1.0);
    uint64_t n = keys.size();

    vector<Node*> leaves;
    Node* node = create_node(true);
    for (uint64_t i = 0; i < n; i++) {
        auto k = keys[i];
        Node* value = (Node*)i;
        bool full = node->count >= Node::maxEntries;

        full = full || static_cast<float>(node->count) / static_cast<float>(Node::maxEntries) > loadFactor;
//	std::cout << "current load: " << static_cast<float>(node->count) / static_cast<float>(Node::maxEntries) << std::endl;
        if (full) {
            leaves.push_back(node);
            node = create_node(true);
            bool inserted = append_into(node, k, value);
            (void)inserted;
            assert(inserted);
            //leaves.back()->upperOrNext = node;
        } else {
            bool appended = append_into(node, k, value);
            (void)appended;
            assert(appended);
        }
    }
    leaves.push_back(node);

    cout << "count per leaf node: " << n / leaves.size() << endl;

    Node* root = construct_inner_nodes(leaves, loadFactor);
    return root;
}

Node* construct_dense(uint32_t numElements, float loadFactor) {
    std::vector<uint32_t> keys(numElements);
    std::iota(keys.begin(), keys.end(), 0);
    return construct(keys, loadFactor);
}

static unsigned lower_bound(Node* node, key_t key) {
    //cout << "search key: " << key << " in [" << node->keys[0] << ", " << node->keys[node->count - 1] << "]" << endl;
    unsigned lower = 0;
    unsigned upper = node->count;
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

bool lookup(Node* tree, key_t key, payload_t& result) {
    Node* node = tree;
    while (!node->isLeaf) {
        unsigned pos = lower_bound(node, key);
        //cout << "inner pos: " << pos << endl;
        node = reinterpret_cast<Node*>(node->payloads[pos]);
        if (node == nullptr) {
            return false;
        }
    }

    unsigned pos = lower_bound(node, key);
    //cout << "pos: " << pos << endl;
    if ((pos < node->count) && (node->keys[pos] == key)) {
        result = node->payloads[pos];
        return true;
    }

    return false;
}

#if 0
// can't be compiled by nvcc...
void prefetchTree(Node* tree, int device) {
    if (device < 0) {
        cudaGetDevice(&device);
    }

    const auto prefetchNode = [&](const auto& self, Node* node) -> void {
        cudaMemAdvise(node, Node::pageSize, cudaMemAdviseSetReadMostly, device);
        cudaMemPrefetchAsync(node, btree::Node::pageSize, device);

        if (node->isLeaf) return;
        for (unsigned i = 0; i <= node->count; ++i) {
            Node* child = reinterpret_cast<Node*>(node->payloads[i]);
            assert(child);
            self(self, child);
        }
    };
    prefetchNode(prefetchNode, tree);
}
#endif

void prefetchSubtree(Node* node, int device) {
    bool isLeaf = node->isLeaf;

    if (!isLeaf) {
        for (unsigned i = 0; i <= node->count; ++i) {
            Node* child = reinterpret_cast<Node*>(node->payloads[i]);
            assert(child);
            prefetchSubtree(child, device);
        }
    }

    cudaMemPrefetchAsync(node, btree::Node::pageSize, device);

    if (isLeaf) return;
};

void prefetchTree(Node* tree, int device) {
    if (device < 0) {
        cudaGetDevice(&device);
    }
    prefetchSubtree(tree, device);
    cudaDeviceSynchronize();
}

namespace cuda {

__device__ unsigned naive_lower_bound(Node* node, key_t key) {
    unsigned lower = 0;
    unsigned upper = node->count;
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

__device__ payload_t btree_lookup(Node* tree, key_t key) {
    //printf("btree_lookup key: %lu\n", key);
    Node* node = tree;
    while (!node->isLeaf) {
        unsigned pos = naive_lower_bound(node, key);
        //printf("inner pos: %d\n", pos);
        node = reinterpret_cast<Node*>(node->payloads[pos]);
        if (node == nullptr) {
            return invalidTid;
        }
    }

    unsigned pos = naive_lower_bound(node, key);
    //printf("leaf pos: %d\n", pos);
    if ((pos < node->count) && (node->keys[pos] == key)) {
        return node->payloads[pos];
    }

    return invalidTid;
}

__device__ unsigned lower_bound_with_hint(Node* node, key_t key, float hint) {
    return 0;
}

__global__ void btree_bulk_lookup(Node* tree, unsigned n, uint32_t* keys, payload_t* tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //printf("index: %d stride: %d\n", index, stride);
    for (int i = index; i < n; i += stride) {
        tids[i] = btree_lookup(tree, keys[i]);
        //printf("tids[%d] = %lu\n", tids[i]);
    }
}

} // namespace cuda

} // namespace btree
