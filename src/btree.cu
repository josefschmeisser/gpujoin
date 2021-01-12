#include "btree.cuh"

#include <cstddef>
#include <cstdio>
#include <vector>
#include <iostream>
#include <tuple>
#include <cassert>
#include <numeric>
#include <limits>

#include <numa.h>

using namespace std;

namespace btree {

Node* create_node(bool isLeaf) {
    Node* node;
    void** dst = reinterpret_cast<void**>(&node);
    cudaMallocManaged(dst, Node::pageSize);
    //node = reinterpret_cast<Node*>(numa_alloc_onnode(Node::pageSize, 0));
    node->isLeaf = isLeaf;

    // initialize key vector with the largest key value possible
    static constexpr auto maxKey = std::numeric_limits<key_t>::max();
    for (unsigned i = 0; i < Node::maxEntries; ++i) {
        node->keys[i] = maxKey;
    }

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
            node->payloads[node->count] = reinterpret_cast<payload_t>(curr);
            currentLevel.push_back(node);
            node = create_node(false);
        } else {
            bool appended = append_into(node, sep, reinterpret_cast<payload_t>(curr));
            (void)appended;
            assert(appended);
        }
    }
    //node->upperOrNext = lowerLevel[lowerLevel.size() - 1];
    node->payloads[node->count] = reinterpret_cast<payload_t>(lowerLevel[lowerLevel.size() - 1]);
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
        payload_t value = i;
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

    cout << "tree size: " << tree_size_in_byte(root) / (1024*1024) << " MB" << endl;
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

size_t tree_size_in_byte(Node* tree) {
    if (tree->isLeaf) { return Node::pageSize; }

    size_t size = Node::pageSize;
    for (unsigned i = 0; i <= tree->count; ++i) {
        Node* child = reinterpret_cast<Node*>(tree->payloads[i]);
        assert(child);
        size += tree_size_in_byte(child);
    }
    return size;
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

template<class T>
__device__ unsigned branchy_binary_search(T x, const T* arr, unsigned size) {
    unsigned lower = 0;
    unsigned upper = size;
    do {
        unsigned mid = ((upper - lower) / 2) + lower;
        if (x < arr[mid]) {
            upper = mid;
        } else if (x > arr[mid]) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}

template<class T>
__device__ unsigned branch_free_binary_search(T x, const T* arr, unsigned size) {
//    if (size < 1) { return 0; }

    const unsigned steps = 31 - __clz(size - 1);
    //printf("steps: %d\n", steps);
    unsigned mid = 1 << steps;

    unsigned ret = (arr[mid] < x) * (size - mid);
    //while (mid > 0) {
    for (unsigned step = 1; step <= steps; ++step) {
        mid >>= 1;
        ret += (arr[ret + mid] < x) ? mid : 0;
    }
    ret += (arr[ret] < x) ? 1 : 0;

    return ret;
}

template<class T, unsigned max_step = 8> // TODO find optimal limit
__device__ unsigned branch_free_exponential_search(T x, const T* arr, unsigned n, float hint) {
//    if (size < 1) { return 0; }

    const int last = n - 1;
    const int start = static_cast<int>(last*hint);
    assert(start <= last);

    bool cont = true;
    bool less = arr[start] < x;
    int offset = -1 + 2*less;
    unsigned current = max(0, min(last , start + offset));
    for (unsigned i = 0; i < max_step; ++i) {
        cont = ((arr[current] < x) == less);
        offset = cont ? offset<<1 : offset;
        current = max(0, min(last , start + offset));
    }

    const auto pre_lower = max(0, min(static_cast<int>(n), start + (offset>>less)));
    const auto pre_upper = 1 + max(0, min(static_cast<int>(n), start + (offset>>(1 - less))));
    const unsigned lower = (!cont || less) ? pre_lower : 0;
    const unsigned upper = (!cont || !less) ? pre_upper : n;

    return lower + branchy_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
}

__device__ payload_t btree_lookup(Node* tree, key_t key) {
    //printf("btree_lookup key: %lu\n", key);
    Node* node = tree;
    while (!node->isLeaf) {
        //unsigned pos = naive_lower_bound(node, key);
        unsigned pos = branch_free_binary_search(key, node->keys, node->count);
        //printf("inner pos: %d\n", pos);
        node = reinterpret_cast<Node*>(node->payloads[pos]);/*
        if (node == nullptr) {
            return invalidTid;
        }*/
    }

    //unsigned pos = naive_lower_bound(node, key);
    unsigned pos = branch_free_binary_search(key, node->keys, node->count);
    //printf("leaf pos: %d\n", pos);
    if ((pos < node->count) && (node->keys[pos] == key)) {
        return node->payloads[pos];
    }

    return invalidTid;
}

__device__ payload_t btree_lookup_with_hints(Node* tree, key_t key) {
    //printf("btree_lookup key: %lu\n", key);
    float hint = 0.5f;
    Node* node = tree;
    while (!node->isLeaf) {
        unsigned pos = branch_free_exponential_search(key, node->keys, node->count, hint);
        if (pos > 0) {
            const auto prev = static_cast<float>(node->keys[pos - 1]);
            const auto current = static_cast<float>(node->keys[pos]);
            printf("pref: %f current: %f\n", prev, current);
            hint = (static_cast<float>(key) - prev)/(current - prev);
        } else {
            hint = 0.5f;
        }

        node = reinterpret_cast<Node*>(node->payloads[pos]);
        /*
        if (node == nullptr) {
            return invalidTid;
        }*/
    }

    //unsigned pos = naive_lower_bound(node, key);
    unsigned pos = branch_free_exponential_search(key, node->keys, node->count, hint);
    //printf("leaf pos: %d\n", pos);
    if ((pos < node->count) && (node->keys[pos] == key)) {
        return node->payloads[pos];
    }

    return invalidTid;
}

__global__ void btree_bulk_lookup(Node* tree, unsigned n, uint32_t* keys, payload_t* tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        //tids[i] = btree_lookup(tree, keys[i]);
        //auto tid1 = btree_lookup(tree, keys[i]);
        tids[i] = btree_lookup_with_hints(tree, keys[i]);
        /*
        
        auto tid2 = btree_lookup_with_hints(tree, keys[i]);
        
        if (tid1 != tid2) {
            printf("mismatch\n");
        }

        tids[i] = tid1;*/
   //     printf("tids[%d] = %lu\n", tids[i]);
    }
}

} // namespace cuda

} // namespace btree
