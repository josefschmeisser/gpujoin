#include "tpch_14_common.cuh"

#include <cstdio>
#include <map>
#include <string>
#include <memory>

#include "config.hpp"
#include "indexes.cuh"
#include "utils.hpp"
#include "common.hpp"

struct query_data {
    Database db;

    std::unique_ptr<abstract_index<indexed_t>> index_structure;

    unsigned lineitem_size;
    lineitem_table_plain_t* lineitem_device;
    std::unique_ptr<lineitem_table_plain_t> lineitem_device_ptrs;

    unsigned part_size;
    part_table_plain_t* part_device;
    std::unique_ptr<part_table_plain_t> part_device_ptrs;

    void load_database() {
        auto& config = get_experiment_config();

        load_tables(db, config.db_path);
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
            index_structure = std::make_unique<btree_type>();
        } else if (config.index_type == "harmonia") {
            index_structure = std::make_unique<harmonia_type>();
        } else if (config.index_type == "lower_bound") {
            index_structure = std::make_unique<lower_bound_type>();
        } else if (config.index_type == "radixspline") {
            index_structure = std::make_unique<radix_spline_type>();
        } else if (config.index_type == "no_op") {
            index_structure = std::make_unique<no_op_type>();
        } else {
            assert(false);
        }

        const auto view = make_vector_view(db.part.p_partkey);
        index_structure->construct(view, part_device_ptrs->p_partkey);
        printf("index size: %lu bytes\n", index_structure->memory_consumption());
    }
};


struct abstract_approach_dispatcher {
    virtual void run(query_data& d, index_type_enum index_type) const = 0;
};

template<template<class T> class Func>
struct approach_dispatcher : public abstract_approach_dispatcher {
    void run(query_data& d, index_type_enum index_type) const override {
        printf("run\n");
        switch (index_type) {
            case index_type_enum::btree:
                Func<bool>()(d);
                break;
            default:
                assert(false);
        }
    }
};

template<class IndexType>
struct my_approach {
    void operator()(query_data& d) {
        printf("my approach %s\n", type_name<IndexType>::value());
    }
};

template<class IndexStructureType>
__global__ void test_kernel();

template<class IndexType>
struct streamed_approach {
    void operator()(query_data& d) {
        using namespace std;

        enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

#if 0
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*4; // TODO

        auto left = left_pipeline(db);

        uint32_t* d_l_partkey_buffer = left.l_partkey_buffer_guard.data();
        int64_t* d_l_extendedprice_buffer = left.l_extendedprice_buffer_guard.data();
        int64_t* d_l_discount_buffer = left.l_discount_buffer_guard.data();

        const auto kernelStart = std::chrono::high_resolution_clock::now();

        ij_join_finalization_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(d_l_partkey_buffer, d_l_extendedprice_buffer, d_l_discount_buffer, left.size, part_device, part_size, index_structure.device_index);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
#endif

        //ij_join_streamed_btree<<<1, 32>>>(d.index_structure->device_index);
        test_kernel<IndexType><<<1, 32>>>();
        cudaDeviceSynchronize();
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
    { "option1", std::make_shared<approach_dispatcher<my_approach>>() },
    { "streamed", std::make_shared<approach_dispatcher<streamed_approach>>() }
};

// TODO move to cpp
void execute_approach(std::string approach) {

    query_data qd;/*
    qd.load_database();
    qd.create_index();
*/

    approaches.at(approach)->run(qd, index_type_enum::btree);
}
