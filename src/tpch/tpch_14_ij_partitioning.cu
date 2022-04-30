#include "tpch_14_ij_partitioning.cuh"

#include "common.hpp"
#include "config.hpp"
#include "indexes.cuh"
#include "tpch_14_common.cuh"
#include "cuda_utils.cuh"

#include <prefix_scan_state.h>
#include <gpu_radix_partition.h>

__global__ void partitioned_ij_scan(partitioned_ij_scan_args args) {

    const auto* __restrict__ l_shipdate = args.lineitem->l_shipdate;
    const auto* __restrict__ l_extendedprice = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount = args.lineitem->l_discount;
    const auto* __restrict__ l_partkey = args.lineitem->l_partkey;

    const unsigned my_lane = lane_id();

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.lineitem_size + 31; i += stride) {
        bool active = i < args.lineitem_size;

        // evaluate predicate
        active &= (l_shipdate[i] >= lower_shipdate) && (l_shipdate[i] < upper_shipdate);

        indexed_t partkey;
        payload_t summand;
        if (active) {
            partkey = l_partkey[i];
            summand = l_extendedprice[i] * (100 - l_discount[i]);
        }

        // determine all threads with matching tuples
        uint32_t mask = __ballot_sync(FULL_MASK, active);
        const auto count = __popc(mask);
        if (count < 1) continue;

        // update global buffer index
        const uint32_t right = __funnelshift_l(FULL_MASK, 0, my_lane);
        const unsigned thread_offset = __popc(mask & right);
        uint32_t base = 0;
        if (my_lane == 0) {
            base = atomicAdd(&args.state->materialized_size, count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        if (active) {
            args.state->l_partkey[base + thread_offset] = partkey;
            args.state->summand[base + thread_offset] = summand;
        }
    }
}

__global__ void partitioned_ij_scan_refill(partitioned_ij_scan_args args) {
    static constexpr unsigned tuples_per_thread = 12;
    static constexpr unsigned per_warp_buffer_size = 32 * tuples_per_thread;

    extern __shared__ uint32_t shared_mem[];

    assert(tuples_per_thread * blockDim.x * sizeof(materialized_tuple) <= 48*1024);
    //if (blockIdx.x == 0 && threadIdx.x == 0) printf("buffer_size %u\n", tuples_per_thread * blockDim.x * sizeof(materialized_tuple));

    const auto* __restrict__ l_shipdate = args.lineitem->l_shipdate;
    const auto* __restrict__ l_extendedprice = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount = args.lineitem->l_discount;
    const auto* __restrict__ l_partkey = args.lineitem->l_partkey;

    const unsigned my_lane = lane_id();
    const unsigned warp_id = (threadIdx.x / 32);

    //static constexpr
    materialized_tuple* tuple_buffer = reinterpret_cast<materialized_tuple*>(shared_mem) + (warp_id * per_warp_buffer_size);
    unsigned buffer_count = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.lineitem_size + 31; i += stride) {
        bool active = i < args.lineitem_size;

        // evaluate predicate
        active &= (l_shipdate[i] >= lower_shipdate) && (l_shipdate[i] < upper_shipdate);

        indexed_t partkey;
        payload_t summand;
        if (active) {
            partkey = l_partkey[i];
            summand = l_extendedprice[i] * (100 - l_discount[i]);
        }

        // determine all threads with matching tuples
        uint32_t mask = __ballot_sync(FULL_MASK, active);
        const auto count = __popc(mask);
        if (count < 1) continue;

        // flush buffer
        if (count + buffer_count > per_warp_buffer_size) {
            // reserve space in global materialization buffer
            uint32_t base;
            if (my_lane == 0) {
                base = atomicAdd(&args.state->materialized_size, buffer_count);
            }
            base = __shfl_sync(FULL_MASK, base, 0);

            // materialize into global buffer
            do {
                const auto flush_count = min(warpSize, buffer_count);
                const bool participate = my_lane < buffer_count;
                if (participate) {
                    const auto& src_tuple = tuple_buffer[buffer_count - 1 - my_lane];
                    args.state->l_partkey[base + my_lane] = src_tuple.l_partkey;
                    args.state->summand[base + my_lane] = src_tuple.summand;
                }

                // decrease buffer_count
                buffer_count -= flush_count;
                base += flush_count;
            } while (buffer_count > 0);

            assert(buffer_count == 0);
        }

        // update warp buffer index
        const uint32_t right_mask = __funnelshift_l(FULL_MASK, 0, my_lane);
        const unsigned thread_offset = __popc(mask & right_mask);
        uint32_t base = buffer_count;
        buffer_count += count;

        if (active) {
            auto& tuple = tuple_buffer[base + thread_offset];
            tuple.summand = summand;
            tuple.l_partkey = partkey;
        }
    }

    // flush buffer
    if (buffer_count > 0) {
        // reserve space in global materialization buffer
        uint32_t base;
        if (my_lane == 0) {
            base = atomicAdd(&args.state->materialized_size, buffer_count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        // materialize into global buffer
        do {
            const auto flush_count = min(warpSize, buffer_count);
            const bool participate = my_lane < buffer_count;
            if (participate) {
                const auto& src_tuple = tuple_buffer[buffer_count - 1 - my_lane];
                args.state->l_partkey[base + my_lane] = src_tuple.l_partkey;
                args.state->summand[base + my_lane] = src_tuple.summand;
            }

            // decrease buffer_count
            buffer_count -= flush_count;
            base += flush_count;
        } while (buffer_count > 0);

        assert(buffer_count == 0);
    }
}

template<class IndexStructureType>
__global__ void partitioned_ij_lookup(const partitioned_ij_lookup_args args, const IndexStructureType index_structure) {
    const char* prefix = "PROMO";
    const auto fanout = 1U << args.radix_bits;

    int64_t numerator = 0;
    int64_t denominator = 0;

    const auto* __restrict__ p_type = args.part->p_type;

    for (uint32_t p = args.task_assignment[blockIdx.x]; p < args.task_assignment[blockIdx.x + 1U]; ++p) {
        const partitioned_tuple_type* __restrict__ relation = reinterpret_cast<const partitioned_tuple_type*>(args.rel) + args.rel_partition_offsets[p];

        const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
        const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < partition_size + 31; i += blockDim.x) {
            const bool active = i < partition_size;
            const partitioned_tuple_type tuple = active ? relation[i] : partitioned_tuple_type();
            const auto part_tid = index_structure.cooperative_lookup(active, tuple.key);

            decltype(materialized_tuple::summand) summand = tuple.value;
            if (part_tid != invalid_tid) {
                denominator += summand;

                const char* type = reinterpret_cast<const char*>(&p_type[part_tid]); // FIXME relies on undefined behavior
                if (device_strcmp(type, prefix, 5) == 0) {
                    numerator += summand;
                }
            } else {
                assert(false);
            }
        }
    }

    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&args.state->global_numerator, (unsigned long long int)numerator);
        atomicAdd((unsigned long long int*)&args.state->global_denominator, (unsigned long long int)denominator);
    }
}

template __global__ void partitioned_ij_lookup<btree_type::device_index_t>(const partitioned_ij_lookup_args args, const btree_type::device_index_t index_structure);
template __global__ void partitioned_ij_lookup<harmonia_type::device_index_t>(const partitioned_ij_lookup_args args, const harmonia_type::device_index_t index_structure);
template __global__ void partitioned_ij_lookup<lower_bound_type::device_index_t>(const partitioned_ij_lookup_args args, const lower_bound_type::device_index_t index_structure);
template __global__ void partitioned_ij_lookup<radix_spline_type::device_index_t>(const partitioned_ij_lookup_args args, const radix_spline_type::device_index_t index_structure);
template __global__ void partitioned_ij_lookup<no_op_type::device_index_t>(const partitioned_ij_lookup_args args, const no_op_type::device_index_t index_structure);
