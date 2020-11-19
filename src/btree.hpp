#pragma once

#include <bits/stdint-uintn.h>
#include <cstdint>
#include <vector>

namespace btree {

using key_t = uint32_t;
using payload_t = void*;

struct NodeBase {
    bool isLeaf;
    uint16_t count;
};

struct Node : public NodeBase {
    static const uint64_t pageSize = 4 * 1024;
    static const uint64_t maxEntries = ((pageSize - sizeof(NodeBase) - sizeof(payload_t)) / (sizeof(key_t) + sizeof(payload_t))) - 1;

    key_t keys[maxEntries];
    payload_t payloads[maxEntries + 1];
};

Node* construct(const std::vector<uint32_t>& keys, float loadFactor);

Node* construct_dense(uint32_t numElem, float loadFactor);

bool lookup(Node* tree, key_t key, payload_t& result);

};
