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
            auto status = map.insert(leader_key, leader_value, ht_group);
            //assert(status == warpcore::Status::none());

            active_mask ^= 1UL << leader;
        }
    }
}

template<class ArgsType>
__global__ void hj_warpcore_probe_kernel_old(const ArgsType args) {
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

template<class ArgsType>
__global__ void hj_warpcore_single_value_probe_kernel(const ArgsType args) {
    namespace cg = cooperative_groups;

    using key_t = typename ArgsType::key_type;
    using index_t = typename decltype(args.map)::index_type;

    auto* __restrict__ tids = args.tids;
    const auto* __restrict__ probe_side_rel = args.probe_side_rel;
    const auto& map = args.map;
    auto ht_group = cg::tiled_partition<map.cg_size()>(cg::this_thread_block());

    const auto limit = (args.probe_side_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (device_size_t i = index; i < limit; i += stride) {
        const bool active = i < args.probe_side_size;
        uint32_t active_mask = ht_group.ballot(active);

        key_t key {};
        if (active) {
            key = probe_side_rel[i];
        }

        while (active_mask) {
            const auto leader = __ffs(active_mask) - 1;
            const auto leader_key = ht_group.shfl(key, leader);
            size_t value;
            auto status = map.retrieve(leader_key, value, ht_group);

            // Status::none() indicates a match
            if (status == warpcore::Status::none() && ht_group.thread_rank() == leader) {
                tids[value] = leader_key;
            }

            active_mask ^= 1UL << leader;
        }
    }
}

// TODO move
__device__ size_t hj_warpcore_result_count = 0;

template<class ArgsType>
__global__ void hj_warpcore_probe_kernel(const ArgsType args) {
    namespace cg = cooperative_groups;

    using key_t = typename ArgsType::key_type;
    using index_t = typename decltype(args.map)::index_type;

    auto* __restrict__ tids = args.tids;
    const auto* __restrict__ probe_side_rel = args.probe_side_rel;
    const auto& map = args.map;
    auto ht_group = cg::tiled_partition<map.cg_size()>(cg::this_thread_block());

    auto f = [&] (auto k, auto v, auto i) {
        // nop
    };

    const auto limit = (args.probe_side_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (device_size_t i = index; i < limit; i += stride) {
        const auto thread_rank = lane_id();
        const bool active = i < args.probe_side_size;
        uint32_t active_mask = ht_group.ballot(active);

        key_t key {};
        if (active) {
            key = probe_side_rel[i];
        }

        device_size_t hit_count = 0;
        while (active_mask) {
            const auto leader = __ffs(active_mask) - 1;
            const auto leader_key = ht_group.shfl(key, leader);
            device_size_t num_value_out = 0;
            map.for_each(f, leader_key, num_value_out, ht_group);
            if (ht_group.thread_rank() == leader) {
                hit_count = num_value_out;
            }
            active_mask ^= 1UL << leader;
        }

        __syncwarp();

        // reduce hit count
        device_size_t full_count = hit_count;
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            full_count += __shfl_down_sync(FULL_MASK, full_count, offset);
        }
        device_size_t base;
        if (thread_rank == 0) {
            base = tmpl_atomic_add(&hj_warpcore_result_count, full_count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        int64_t remaining_count = hit_count;
        uint32_t straggler_mask = 0;
        uint32_t straggler_count = 0;
        while (true) {
            straggler_mask = __ballot_sync(FULL_MASK, remaining_count > 0);
            straggler_count = __popc(straggler_mask);
            if (straggler_count < 4) break;

            // inactive threads remain in this loop to facilitate the ballot operation
            if (remaining_count > 0) {
                const auto flush_rank = __popc(straggler_mask & ((1 << thread_rank) - 1));
                tids[base + flush_rank] = key;
                remaining_count -= 1;
            }
            base += straggler_count;
        }
        while (straggler_mask != 0) {
            const auto leader = __ffs(straggler_mask) - 1;
            const auto leader_key = __shfl_sync(FULL_MASK, key, leader);
            const auto leader_count = __shfl_sync(FULL_MASK, remaining_count, leader);

            for (device_size_t i = thread_rank; i < leader_count; i += 32) {
                tids[base + i] = leader_key;
            }
            base += leader_count;

            straggler_mask ^= 1UL << leader;
        }
    }
}
