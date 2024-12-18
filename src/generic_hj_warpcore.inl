#include <sys/types.h>
#include <cooperative_groups.h>

#include "cuda_utils.cuh"
#include "index_lookup_config.hpp"

template<class ArgsType>
__global__ void hj_warpcore_build_kernel(ArgsType args) {
    namespace cg = cooperative_groups;

    using key_t = typename ArgsType::key_type;

    const auto* __restrict__ build_side_rel = args.build_side_rel;
    auto& map = args.map;
    auto ht_group = cg::tiled_partition<map.cg_size()>(cg::this_thread_block());
    //printf("cg_size: %lu\n", map.cg_size());

    //const auto limit = args.build_side_size;
    const auto limit = (args.build_side_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < limit; i += stride) {
        const bool active = i < args.build_side_size;
        uint32_t active_mask = ht_group.ballot(active);

        key_t key {};
        if (active) {
            key = build_side_rel[i];
        }
        //printf("before k: %lu lane_id: %d\n", key, lane_id());

        while (active_mask) {
            const auto leader = __ffs(active_mask) - 1;
            const auto leader_key = ht_group.shfl(key, leader);
            const auto leader_value = ht_group.shfl(i, leader);
            //if (ht_group.thread_rank() == leader) {
                //printf("inserting k: %lu v: %lu leader: %d\n", leader_key, leader_value, leader);
            //}
            map.insert(leader_key, leader_value, ht_group);
            active_mask ^= 1UL << leader;
        }
    }
}

template<class ArgsType>
__global__ void hj_warpcore_probe_kernel(const ArgsType args) {
    namespace cg = cooperative_groups;

    using key_t = typename ArgsType::key_type;
    using index_t = typename decltype(args.map)::index_type;

    auto* __restrict__ tids = args.tids;
    const auto* __restrict__ probe_side_rel = args.probe_side_rel;
    const auto& map = args.map;
    auto ht_group = cg::tiled_partition<map.cg_size()>(cg::this_thread_block());
    index_t num_value_out;

    auto f = [&] (auto k, auto v, auto i) {
        //printf("got k: %lu v: %lu\n", k, v);
        tids[v] = k;
    };

    //const auto limit = args.probe_side_size;
    const auto limit = (args.probe_side_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (device_size_t i = index; i < limit; i += stride) {
        //printf("probing k: %lu\n", probe_key);

        const bool active = i < args.probe_side_size;
        uint32_t active_mask = ht_group.ballot(active);

        key_t key {};
        if (active) {
            key = probe_side_rel[i];
        }

        while (active_mask) {
            const auto leader = __ffs(active_mask) - 1;
            const auto leader_key = ht_group.shfl(key, leader);
            //if (ht_group.thread_rank() == leader) {
                //printf("probing k: %lu leader: %d\n", leader_key, leader);
            //}
            map.for_each(f, leader_key, num_value_out, ht_group);
            active_mask ^= 1UL << leader;
        }
    }
}
