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
    cudaMallocManaged(&node, Node::pageSize);
    return node;
}

bool append_into(Node* dst, key_t key, payload_t value) {
    if (dst->count == Node::maxEntries) { return false; }

    dst->keys[dst->count] = key;
    dst->payloads[dst->count] = value;
    dst->count += 1;
    return true;
}

Node* construct_inner_nodes(vector<Node*> lowerLevel, float loadFactor) {
    if (lowerLevel.size() == 1) {
        return lowerLevel.front();
    }

    vector<Node*> currentLevel;
    Node* node = create_node(false);
    for (unsigned i = 0; i < lowerLevel.size() - 1; i++) {
        Node* curr = lowerLevel[i];
        Node* next = lowerLevel[i + 1];
        key_t sep = next->keys[0];
        bool full = node->count >= Node::maxEntries;
        full = full || static_cast<float>(node->count) / static_cast<float>(Node::maxEntries) > loadFactor;
        if (full) {
            node->upper_or_next = curr;
            currentLevel.push_back(node);
            node = create_node(false);
        } else {
            bool appended = append_into(node, sep, curr);
            assert(appended);
        }
    }
    node->upper_or_next = lowerLevel[lowerLevel.size() - 1];
    currentLevel.push_back(node);
    cout << "countPerNode:" << lowerLevel.size() / currentLevel.size() << endl;

    return construct_inner_nodes(currentLevel, loadFactor);
}

Node* construct_tree(const vector<key_t>& keys, float loadFactor) {
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
            leaves.back()->upper_or_next = node;
        } else {
            bool appended = append_into(node, k, value);
            assert(appended);
        }
    }
    leaves.push_back(node);

    cout << "countPerNode:" << n / leaves.size() << endl;

    Node* root = construct_inner_nodes(leaves, loadFactor);
    return root;
}

Node* construct_dense(uint32_t numElements, float loadFactor) {
    std::vector<uint32_t> keys(numElements);
    std::iota(keys.begin(), keys.end(), 0);
    return construct_tree(keys, 0.8);
}

};
