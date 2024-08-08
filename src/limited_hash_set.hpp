#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>

template<class Key, class HashFn, class Allocator>
class limited_hash_set {
public:
    static constexpr float max_load_factor = 0.5;
    //static constexpr Key empty_marker = std::numeric_limits<Key>::max();

    using key_t = Key;

    struct iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = key_t;
        using pointer           = key_t*;
        using reference         = key_t&;

        iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() { return *adjust(); }

        pointer operator->() { return adjust(); }

        iterator& operator++() { m_ptr++; return *this; }

        iterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const iterator& a, const iterator& b) { return a.m_ptr == b.m_ptr; };

        friend bool operator!= (const iterator& a, const iterator& b) { return a.m_ptr != b.m_ptr; };

    private:
        pointer adjust() {
            while (true) {
                if (*m_ptr != empty_marker()) {
                    return m_ptr;
                }

                m_ptr += 1;
            }
        }

        pointer m_ptr;
    };

    iterator begin() {
        return iterator(&table[0]);
    }

    iterator end() {
        int64_t last = capacity - 1;
        for (; last >= 0; --last) {
            if (table[last] != empty_marker()) break;
        } 
        return iterator(&table[last + 1]);
    }

    bool insert(Key key) {
        if (key == empty_marker()) {
            throw std::runtime_error("invalid key");
        }

        size_t slot = HashFn{}(key) & (capacity - 1u);
        for (size_t i = 0; i < capacity; ++i) {
            key_t expected = empty_marker();
            // __atomic_compare_exchange (type *ptr, type *expected, type *desired, bool weak, int success_memorder, int failure_memorder)
            __atomic_compare_exchange(&table[slot], &expected, &key, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
            if (expected == key) {
                return false;
            } else if (expected == empty_marker()) {
                return true;
            }
            slot = (slot + 1u) & (capacity - 1u);
        }
        throw std::runtime_error("capacity exceeded");
    }

    bool lookup(Key key) {
        size_t slot = HashFn{}(key) & (capacity - 1u);
        for (size_t i = 0; i < capacity; ++i) {
            key_t current = &table[slot];
            if (current == key) {
                return true;
            } else if (current == empty_marker()) {
                return false;
            }
            slot = (slot + 1) & (capacity - 1);
        }
        return false;
    }

    static constexpr size_t calculate_table_size(size_t occupancy_upper_bound) {
        float size_hint = static_cast<float>(occupancy_upper_bound) / max_load_factor;
        size_t n = static_cast<size_t>(std::ceil(std::log2(size_hint))); // find n such that: 2^n >= size_hint
        size_t table_size = 1u << n;
        return table_size;
    }

    static constexpr key_t empty_marker() {
        return std::numeric_limits<Key>::max();
    }

    limited_hash_set(size_t occupancy_upper_bound) {
        capacity = calculate_table_size(occupancy_upper_bound);
        //printf("table size: %lu\n", capacity);

        using bound_allocator_type = typename Allocator::template rebind<Key>::other;
        static bound_allocator_type allocator;
        table = allocator.allocate(capacity);

        std::fill(table, table + capacity, empty_marker());
    }

    ~limited_hash_set() {
        using bound_allocator_type = typename Allocator::template rebind<Key>::other;
        static bound_allocator_type allocator;
        allocator.deallocate(table, capacity);
    }

private:
    key_t* table = nullptr;
    size_t capacity = 0u;
};
