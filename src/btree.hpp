#pragma once

#include <bits/stdint-uintn.h>
#include <cstdint>

namespace btree {

using key_t = uint32_t;
using payload_t = void*;

struct NodeBase {
    bool isLeaf;
    uint16_t count;
    NodeBase* upperOrNext;
};

struct Node : public NodeBase {
    static const uint64_t pageSize = 4 * 1024;
    static const uint64_t maxEntries = ((pageSize - sizeof(NodeBase)) / (sizeof(key_t) + sizeof(payload_t))) - 1;

    key_t keys[maxEntries];
    payload_t payloads[maxEntries];
};

Node* construct_dense(uint32_t numElem, float loadFactor);

};
