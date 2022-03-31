#include "tpch_14_common.cuh"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <memory>

#include "config.hpp"
#include "common.hpp"
#include "device_properties.hpp"
#include "indexes.cuh"
#include "utils.hpp"
#include "LinearProbingHashTable.cuh"
#include "measuring.hpp"

index_type_enum parse_index_type(const std::string& index_name) {
    if (index_name == "btree") {
        return index_type_enum::btree;
    } else if (index_name == "harmonia") {
        return index_type_enum::harmonia;
    } else if (index_name == "lower_bound") {
        return index_type_enum::lower_bound;
    } else if (index_name == "radix_spline") {
        return index_type_enum::radix_spline;
    } else if (index_name == "no_op") {
        return index_type_enum::no_op;
    } else {
        throw std::runtime_error("unknown index type");
    }
}

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
        const auto& config = get_experiment_config();

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
        const auto& config = get_experiment_config();

        // allocate index structure
        switch (parse_index_type(config.index_type)) {
            case index_type_enum::btree:
                index_structure = std::make_unique<btree_type>();
                break;
            case index_type_enum::harmonia:
                index_structure = std::make_unique<harmonia_type>();
                break;
            case index_type_enum::lower_bound:
                index_structure = std::make_unique<lower_bound_type>();
                break;
            case index_type_enum::radix_spline:
                index_structure = std::make_unique<radix_spline_type>();
                break;
            case index_type_enum::no_op:
                index_structure = std::make_unique<no_op_type>();
                break;
            default:
                assert(false);
        }

        const auto view = make_vector_view(db.part.p_partkey);
        index_structure->construct(view, part_device_ptrs->p_partkey);
        printf("index size: %lu bytes\n", index_structure->memory_consumption());
    }
};

static measuring::experiment_description create_experiment_description() {
    const auto& config = get_experiment_config();

    measuring::experiment_description r;
    r.name = "tpch_query14";
    r.approach = config.approach;
    std::vector<std::pair<std::string, std::string>> other = {
        std::make_pair(std::string("device"), std::string(get_device_properties(0).name)),
        std::make_pair(std::string("index_type"), config.index_type),
        std::make_pair(std::string("db_path"), config.db_path),
        std::make_pair(std::string("prefetch_index"), tmpl_to_string(config.prefetch_index)),
        std::make_pair(std::string("sort_indexed_relation"), tmpl_to_string(config.sort_indexed_relation)),
        std::make_pair(std::string("block_size"), tmpl_to_string(config.block_size)),

        // allocators:
        std::make_pair(std::string("host_allocator"), std::string(type_name<host_allocator<int>>::value())),
        std::make_pair(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value())),
        std::make_pair(std::string("device_table_allocator"), std::string(type_name<device_table_allocator<int>>::value()))
    };
    r.other.swap(other);

    return r;
}

struct abstract_approach_dispatcher {
    virtual void run(query_data& d, index_type_enum index_type) const = 0;
};

template<template<class T> class Func>
struct approach_dispatcher : public abstract_approach_dispatcher {
    void run(query_data& d, index_type_enum index_type) const override {
        switch (index_type) {
            case index_type_enum::btree:
                Func<btree_type>()(d);
                break;
            case index_type_enum::harmonia:
                Func<harmonia_type>()(d);
                break;
            case index_type_enum::lower_bound:
                Func<lower_bound_type>()(d);
                break;
            case index_type_enum::radix_spline:
                Func<radix_spline_type>()(d);
                break;
            case index_type_enum::no_op:
                Func<no_op_type>()(d);
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

using device_ht_t = LinearProbingHashTable<uint32_t, size_t>::DeviceHandle;

__global__ void hj_build_kernel(size_t n, const part_table_plain_t* part, device_ht_t ht);

__global__ void hj_probe_kernel(size_t n, const part_table_plain_t* __restrict__ part, const lineitem_table_plain_t* __restrict__ lineitem, device_ht_t ht);

template<class IndexType>
struct hj_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        LinearProbingHashTable<uint32_t, size_t> ht(d.part_size);
        int num_blocks = (d.part_size + config.block_size - 1) / config.block_size;
        hj_build_kernel<<<num_blocks, config.block_size>>>(d.part_size, d.part_device, ht.deviceHandle);

        //num_blocks = 32*num_sms;
        num_blocks = (d.lineitem_size + config.block_size - 1) / config.block_size;
        hj_probe_kernel<<<num_blocks, config.block_size>>>(d.lineitem_size, d.part_device, d.lineitem_device, ht.deviceHandle);
        cudaDeviceSynchronize();
    }
};

template<class IndexStructureType>
__global__ void test_kernel();

template<class IndexType>
struct streamed_ij_approach {
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

//static const std::map<std::string, std::unique_ptr<abstract_approach_dispatcher>> approaches {
static const std::map<std::string, std::shared_ptr<abstract_approach_dispatcher>> approaches {
    { "option1", std::make_shared<approach_dispatcher<my_approach>>() },
    { "hj", std::make_shared<approach_dispatcher<hj_approach>>() },
    { "streamed_ij", std::make_shared<approach_dispatcher<streamed_ij_approach>>() }
};

void execute_approach(std::string approach) {
    auto& config = get_experiment_config();

    query_data qd;
    qd.load_database();
    if (config.approach != "hj") {
        qd.create_index();
    }

    const auto experiment_desc = create_experiment_description();
    index_type_enum index_type = parse_index_type(config.index_type);
    measure(experiment_desc, [&]() {
        approaches.at(approach)->run(qd, index_type);
    });
}
