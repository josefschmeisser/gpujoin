#include "btree.hpp"

#include <vector>
#include <iostream>
#include <tuple>
#include <cassert>
#include <numeric>

#include <bits/stdint-uintn.h>
#include <cuda_runtime_api.h>

using namespace std;

namespace btree {

Node* create_node(bool isLeaf) {
    Node* node;
    void** dst = reinterpret_cast<void**>(&node);
    cudaMallocManaged(dst, Node::pageSize);
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
    cout << "search key: " << key << " in [" << node->keys[0] << ", " << node->keys[node->count - 1] << "]" << endl;
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
        cout << "inner pos: " << pos << endl;
        node = reinterpret_cast<Node*>(node->payloads[pos]);
        if (node == nullptr) {
            return false;
        }
    }

    unsigned pos = lower_bound(node, key);
    cout << "pos: " << pos << endl;
    if ((pos < node->count) && (node->keys[pos] == key)) {
        result = node->payloads[pos];
        return true;
    }

    return false;
}

};
