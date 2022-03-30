#pragma once

#include <cstdio>
#include <map>
#include <string>
#include <memory>

#include "config.hpp"
#include "indexes.cuh"
#include "utils.hpp"
#include "common.hpp"

// allocators:
template<class T> using host_allocator = HOST_ALLOCATOR_TYPE;
template<class T> using device_index_allocator = DEVICE_INDEX_ALLOCATOR_TYPE;
template<class T> using device_table_allocator = DEVICE_RELATION_ALLOCATOR;

using indexed_t = std::remove_pointer_t<decltype(lineitem_table_plain_t::l_partkey)>;
using payload_t = uint32_t;

struct query_data {
    Database db;

    std::unique_ptr<abstract_index<indexed_t>> index_structure;

    unsigned lineitem_size;
    lineitem_table_plain_t* lineitem_device;
    std::unique_ptr<lineitem_table_plain_t> lineitem_device_ptrs;

    unsigned part_size;
    part_table_plain_t* part_device;
    std::unique_ptr<part_table_plain_t> part_device_ptrs;

    void load_database(const std::string& path) {
        auto& config = get_experiment_config();

        load_tables(db, path);
        if (config.sort_indexed_relation) {
            printf("sorting part relation...\n");
            sort_relation(db.part);
        }
        lineitem_size = db.lineitem.l_orderkey.size();
        part_size = db.part.p_partkey.size();

        {
            using namespace std;
            const auto start = chrono::high_resolution_clock::now();
            device_table_allocator<int> allocator;
            std::tie(lineitem_device, lineitem_device_ptrs) = migrate_relation(db.lineitem, allocator);
            std::tie(part_device, part_device_ptrs) = migrate_relation(db.part, allocator);
            const auto finish = chrono::high_resolution_clock::now();
            const auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
            std::cout << "migration time: " << d << " ms\n";
        }
    }

    void create_index() {
        auto& config = get_experiment_config();
        if (config.index_type == "btree") {
            index_structure = std::make_unique<btree_index<indexed_t, payload_t, device_index_allocator, host_allocator>>();
        } else {
            assert(false);
        }

        const auto view = make_vector_view(db.part.p_partkey);
        index_structure->construct(view, part_device_ptrs->p_partkey);
        printf("index size: %lu bytes\n", index_structure->memory_consumption());
    }
};


struct abstract_approach_dispatcher {
    virtual void run(index_type_enum index_type) const = 0;
};

template<template<class T> class Func>
struct approach_dispatcher : public abstract_approach_dispatcher {
    void run(index_type_enum index_type) const override {
        printf("run\n");
        switch (index_type) {
            case index_type_enum::btree:
                Func<bool>()();
                break;
            default:
                assert(false);
        }
    }
};

template<class IndexType>
struct my_approach {
    void operator()() {
        printf("my approach %s\n", type_name<IndexType>::value());
    }
};

/*
static const std::map<std::string, Option> optionStrings {
    { "option1", Option1 },
    { "option2", Option2 },
    //...
};*/
//static const std::map<std::string, std::unique_ptr<abstract_approach_dispatcher>> approaches {
static const std::map<std::string, std::shared_ptr<abstract_approach_dispatcher>> approaches {
    { "option1", std::make_shared<approach_dispatcher<my_approach>>() }//,
    //{ "option2", Option2 },
    //...
};

void run_approach(std::string approach) {
    /*
    query_data qd;
    qd.load_database();
    qd.create_index();
*/
    const auto& test = approaches.at(approach);
    approaches.at(approach)->run(index_type_enum::btree);
/*
  approach_dispatcher<my_approach> test;
  test.run(index_type_enum::btree);*/
}
