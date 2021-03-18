#include "btree.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <tuple>
#include <cassert>
#include <numeric>
#include <limits>
#include <memory>

#include <numa.h>

using namespace std;

static const size_t cache_line_size = 128;

namespace btree {

Node* create_node(bool isLeaf) {
    Node* node;
    void** dst = reinterpret_cast<void**>(&node);
    cudaMallocManaged(dst, Node::pageSize);
    //node = reinterpret_cast<Node*>(numa_alloc_onnode(Node::pageSize, 0));
    node->header.isLeaf = isLeaf;

    // validate alignment
    if ((reinterpret_cast<uintptr_t>(node) & cache_line_size-1) != 0) { throw std::runtime_error("unaligned memory"); }
    if ((reinterpret_cast<uintptr_t>(&node->keys[0]) & cache_line_size-1) != 0) { throw std::runtime_error("unaligned memory"); }

    // initialize key vector with the largest key value possible
    static constexpr auto maxKey = std::numeric_limits<key_t>::max();
    for (unsigned i = 0; i < Node::maxEntries; ++i) {
        node->keys[i] = maxKey;
    }

    return node;
}

bool append_into(Node* dst, key_t key, payload_t value) {
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

Node* construct_inner_nodes(vector<Node*> lowerLevel, float loadFactor) {
    if (lowerLevel.size() == 1) {
        return lowerLevel.front();
    }

    vector<Node*> currentLevel;
    Node* node = create_node(false);
    for (unsigned i = 0; i < lowerLevel.size() - 1; i++) {
        Node* curr = lowerLevel[i];
        key_t sep = max_key(curr);
        bool full = node->header.count >= Node::maxEntries;
        full = full || static_cast<float>(node->header.count) / static_cast<float>(Node::maxEntries) > loadFactor;
        if (full) {
            //node->upperOrNext = curr;
            node->payloads[node->header.count] = reinterpret_cast<payload_t>(curr);
            currentLevel.push_back(node);
            node = create_node(false);
        } else {
            bool appended = append_into(node, sep, reinterpret_cast<payload_t>(curr));
            (void)appended;
            assert(appended);
        }
    }
    //node->upperOrNext = lowerLevel[lowerLevel.size() - 1];
    node->payloads[node->header.count] = reinterpret_cast<payload_t>(lowerLevel[lowerLevel.size() - 1]);
    currentLevel.push_back(node);
    cout << "header.count per inner node: " << lowerLevel.size() / currentLevel.size() << endl;

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
        bool full = node->header.count >= Node::maxEntries;
        full = full || static_cast<float>(node->header.count) / static_cast<float>(Node::maxEntries) > loadFactor;
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

    cout << "header.count per leaf node: " << n / leaves.size() << endl;

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

bool lookup(Node* tree, key_t key, payload_t& result) {
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

#if 0
// can't be compiled by nvcc...
void prefetchTree(Node* tree, int device) {
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

void prefetchSubtree(Node* node, int device) {
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

void prefetchTree(Node* tree, int device) {
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

namespace cuda {

template<class T>
__device__ unsigned branchy_binary_search(T x, const T* arr, const unsigned size) {
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
__device__ unsigned branch_free_binary_search(T x, const T* arr, const unsigned size) {
    if (size < 1) { return 0; }

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

template<class T, unsigned max_step = 4> // TODO find optimal limit
__device__ unsigned branch_free_exponential_search(T x, const T* arr, const unsigned n, const float hint) {
    //if (size < 1) return;

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
        current = max(0, min(last, start + offset));
    }

    const auto pre_lower = max(0, min(static_cast<int>(n), start + (offset>>less)));
    const auto pre_upper = 1 + max(0, min(static_cast<int>(n), start + (offset>>(1 - less))));
    const unsigned lower = (!cont || less) ? pre_lower : 0;
    const unsigned upper = (!cont || !less) ? pre_upper : n;

    return lower + branchy_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
//    return lower + branch_free_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
}

template<class T>
__device__ unsigned exponential_search(T x, const T* arr, const unsigned size) {
    assert(size > 0);
    int bound = 1;
    while (bound < size && arr[bound] < x) {
        bound <<= 1;
    }
    const auto lower = bound>>1;
    return lower + branchy_binary_search(x, arr + lower, min(bound + 1, size - lower));
}

template<class T>
__device__ unsigned linear_search(T x, const T* arr, const unsigned size) {
    for (unsigned i = 0; i < size; ++i) {
        if (arr[i] >= x) return i;
    }
    return size;
}

__device__ payload_t btree_lookup(const Node* tree, key_t key) {
    //printf("btree_lookup key: %lu\n", key);
    const Node* node = tree;
    while (!node->header.isLeaf) {
        //unsigned pos = branchy_binary_search(key, node->keys, node->header.count);
        unsigned pos = linear_search(key, node->keys, node->header.count);
        //printf("inner pos: %d\n", pos);
        node = reinterpret_cast<const Node*>(node->payloads[pos]);/*
        if (node == nullptr) {
            return invalidTid;
        }*/
    }

    //unsigned pos = branchy_binary_search(key, node->keys, node->header.count);
    unsigned pos = linear_search(key, node->keys, node->header.count);
    //printf("leaf pos: %d\n", pos);
    if ((pos < node->header.count) && (node->keys[pos] == key)) {
        return node->payloads[pos];
    }

    return invalidTid;
}

#if 1
__device__ payload_t btree_lookup_with_hints(const Node* tree, key_t key) {
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
            return invalidTid;
        }*/
    }

    //unsigned pos = naive_lower_bound(node, key);
    unsigned pos = branch_free_exponential_search(key, node->keys, node->header.count, hint);
    //printf("leaf pos: %d\n", pos);
/*
    if ((pos < node->header.count) && (node->keys[pos] == key)) {
        return node->payloads[pos];
    }
    return invalidTid;
*/
    return (pos < node->header.count) && (node->keys[pos] == key) ? node->payloads[pos] : invalidTid;
}
#endif

#if 0
__device__ payload_t btree_lookup_with_hints(const Node* tree, key_t key) {
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
    return (pos < node->header.count) && (node->keys[pos] == key) ? node->payloads[pos] : invalidTid;
}
#endif

#if 0
__device__ payload_t btree_lookup_with_hints(const Node* tree, key_t key) {
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

    return (pos < node->header.count) && (node->keys[pos] == key) ? node->payloads[pos] : invalidTid;
}
#endif

#if 0
__device__ payload_t btree_lookup_with_hints(const Node* tree, key_t key) {
    //printf("btree_lookup key: %lu\n", key);
    float hint = 0.5;
    const Node* node;
    unsigned pos;
    payload_t payload = reinterpret_cast<payload_t>(tree);
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

    return invalidTid;
}
#endif

} // namespace cuda

} // namespace btree
