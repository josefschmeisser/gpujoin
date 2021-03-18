#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>

namespace btree {

using key_t = uint32_t;
using payload_t = uintptr_t;

static constexpr payload_t invalidTid = std::numeric_limits<btree::payload_t>::max();

/*
struct NodeBase {
    bool isLeaf;
    uint16_t count;
};
*/
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
    static const uint64_t maxEntries = ((pageSize - sizeof(NodeBase) - sizeof(payload_t)) / (sizeof(key_t) + sizeof(payload_t))) - 1;

    key_t keys[maxEntries];
    payload_t payloads[maxEntries + 1];
};
//static_assert(offsetof(Node, keys) == 128);
static_assert(sizeof(Node) < Node::pageSize);

Node* construct(const std::vector<uint32_t>& keys, float loadFactor);

Node* construct_dense(uint32_t numElem, float loadFactor);

bool lookup(Node* tree, key_t key, payload_t& result);

void prefetchTree(Node* tree, int device = -1);

size_t tree_size_in_byte(Node* tree);

namespace cuda {

__device__ payload_t btree_lookup(const Node* tree, key_t key);

__device__ payload_t btree_lookup_with_hints(const Node* tree, key_t key);

//__global__ void btree_bulk_lookup(const Node* tree, unsigned n, uint32_t* keys, payload_t* tids);

} // namespace cuda

} // namespace btree
