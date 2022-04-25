#include "tpch_14_streamed_ij.cuh"

#include "common.hpp"
#include "config.hpp"


#include "indexes.cuh"
//#include "device_array.hpp"

#include "tpch_14_common.cuh"

//#include "prefix_scan_state.h"
#include <prefix_scan_state.h>


//#include <numa-gpu/sql-ops/include/prefix_scan.h>
#include <numa-gpu/sql-ops/include/gpu_common.h>

#include <gpu_radix_partition.h>

#include "cuda_utils.cuh"



/*
TODO
*/



// Exports the vanilla index join kernel for 8-byte keys.
extern "C" __launch_bounds__(1024, 1) __global__ void ij_join_streamed_btree(const btree_type::device_index_t index) {
    printf("from gpu\n");
}




template<class IndexStructureType>
__global__ void test_kernel() {
    printf("from gpu\n");
}

template __global__ void test_kernel<btree_type>();
template __global__ void test_kernel<harmonia_type>();
template __global__ void test_kernel<lower_bound_type>();
template __global__ void test_kernel<radix_spline_type>();
template __global__ void test_kernel<no_op_type>();

#if 0
struct materialized_tuple {
    std::remove_pointer<decltype(lineitem_table_plain_t::l_extendedprice)>::type summand;
    //decltype(*lineitem_table_plain_t::l_extendedprice) l_extendedprice;
    //decltype(*lineitem_table_plain_t::l_discount) l_discount;
    std::remove_pointer<decltype(lineitem_table_plain_t::l_partkey)>::type l_partkey;
};

struct partitioned_ij_scan_args {
    // Input
    const lineitem_table_plain_t lineitem;
    const size_t lineitem_size;
    const part_table_plain_t part;
    const void* index_structure;

    std::size_t const canonical_chunk_length; // TODO needed?
    uint32_t const padding_length;
    uint32_t const radix_bits;
    uint32_t const ignore_bits;
    // State
    /*
    std::tuple<
        decltype(lineitem.l_extendedprice),
        decltype(lineitem.l_discount),
        decltype(lineitem.l_partkey)
        >* __restrict__ materialized;*/
    materialized_tuple* __restrict__ materialized;
    uint32_t* materialized_size;

    ScanState<unsigned long long> *const prefix_scan_state;
    unsigned long long *const __restrict__ tmp_partition_offsets;



    // Output
    int64_t* global_numerator;
    int64_t* global_denominator;
};
#endif

//template<class IndexStructureType>
//__global__ void partitioned_ij_scan(const lineitem_table_plain_t* __restrict__ lineitem, const unsigned lineitem_size, const part_table_plain_t* __restrict__ part, IndexStructureType index_structure) {
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
        active = (l_shipdate[i] >= lower_shipdate) && (l_shipdate[i] < upper_shipdate);

        const auto summand = l_extendedprice[i] * (100 - l_discount[i]);
        const auto partkey = l_partkey[i];

        // TODO materialize

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
//            printf("lane %u store to: %u\n", my_lane, base + offset);
/*
            auto& tuple = args.materialized[base + thread_offset];
            tuple.summand = summand;
            tuple.l_partkey = partkey;*/
            args.state->l_partkey[base + thread_offset] = partkey;
            args.state->summand[base + thread_offset] = summand;
        }
    }

    // TODO compute prefix sum
}

__global__ void partitioned_ij_scan_refill(partitioned_ij_scan_args args) {
    static constexpr unsigned tuples_per_thread = 2;
    //static constexpr unsigned per_warp_buffer_size = 32 * (tuples_per_thread + 1);
    static constexpr unsigned per_warp_buffer_size = 32 * tuples_per_thread;

    extern __shared__ uint32_t shared_mem[];

    const auto* __restrict__ l_shipdate = args.lineitem->l_shipdate;
    const auto* __restrict__ l_extendedprice = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount = args.lineitem->l_discount;
    const auto* __restrict__ l_partkey = args.lineitem->l_partkey;

    const unsigned my_lane = lane_id();

    const unsigned warp_id = (threadIdx.x / 32);

    //static constexpr 
    //decltype(args.lineitem.l_partkey)* partkey_buffer = reinterpret_cast<decltype(args.lineitem.l_partkey)*>(shared_mem) + (warp_id * 32);
    materialized_tuple* tuple_buffer = reinterpret_cast<materialized_tuple*>(shared_mem) + (warp_id * per_warp_buffer_size);
    unsigned buffer_count = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.lineitem_size + 31; i += stride) {
        bool active = i < args.lineitem_size;

        // evaluate predicate
        active = (l_shipdate[i] >= lower_shipdate) && (l_shipdate[i] < upper_shipdate);

        const auto summand = l_extendedprice[i] * (100 - l_discount[i]);
        const auto partkey = l_partkey[i];

        // TODO materialize

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
            for (unsigned j = 0; j < tuples_per_thread; ++j) {
                unsigned thread_offset = j * (tuples_per_thread * 32) + my_lane;
                const auto& src_tuple = tuple_buffer[thread_offset];/*
                auto& dst_tuple = args.materialized[base + thread_offset];
                dst_tuple.summand = src_tuple.summand;
                dst_tuple.l_partkey = src_tuple.summand;*/
				args.state->l_partkey[base + thread_offset] = src_tuple.l_partkey;
				args.state->summand[base + thread_offset] = src_tuple.summand;
            }

            // reset buffer count
            buffer_count = 0;
        }

        // update warp buffer index
        const uint32_t right_mask = __funnelshift_l(FULL_MASK, 0, my_lane);
        const unsigned thread_offset = __popc(mask & right_mask);
        uint32_t base = warp_id * per_warp_buffer_size;
        if (my_lane == 0) {
            base += atomicAdd(&buffer_count, count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        if (active) {
            auto& tuple = tuple_buffer[base + thread_offset];
            tuple.summand = summand;
            tuple.l_partkey = partkey;
        }
    }

    // TODO compute prefix sum
}

/*
template<class TupleType, class IndexStructureType>
__global__ void partitioned_lookup_kernel(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
    const auto fanout = 1U << args.radix_bits;

    for (uint32_t p = args.task_assignment[blockIdx.x]; p < args.task_assignment[blockIdx.x + 1U]; ++p) {
        const TupleType* __restrict__ relation = reinterpret_cast<const TupleType*>(args.rel) + args.rel_partition_offsets[p];

        const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
        const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

#if 0
        // standard lookup implementation
        for (uint32_t i = threadIdx.x; i < partition_size; i += blockDim.x) {
            const TupleType tuple = relation[i];
            const auto tid = index_structure.lookup(tuple.key);
            args.tids[tuple.value] = tid;
        }
#else
        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < partition_size + 31; i += blockDim.x) {
            const bool active = i < partition_size;
            const TupleType tuple = active ? relation[i] : TupleType();
            const auto tid = index_structure.cooperative_lookup(active, tuple.key);
            if (active) {
                args.tids[tuple.value] = tid;
            }
        }
#endif
    }
}
struct partitioned_ij_lookup_args {
    // Inputs
    const part_table_plain_t part;
    decltype(lineitem_table_plain_t::l_partkey) const __restrict__ l_partkey;
    decltype(lineitem_table_plain_t::l_extendedprice) const __restrict__ summand;
    const uint32_t materialized_size;
    // State and outputs
	partitioned_ij_lookup_mutable_state* const state;
};*/

template<class IndexStructureType>
__global__ void partitioned_ij_lookup(const partitioned_ij_lookup_args args, const IndexStructureType index_structure) {
	//using tuple_type = Tuple<indexed_t, payload_t>;

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

#if 0
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_int32(PrefixSumArgs args);

extern "C" __launch_bounds__(1024, 2) __global__ void intermediate_prefix_sum_int32(partitioned_ij_scan_args args) {
    // We keep the PrefixSumArgs in shared memory in order to avoid costly global memory derefenciations.
    // The only reason why we do not directly pass this struct during kernel invocation is that we have to derefence the size field of partitioned_ij_scan_args.
 /*
    __shared__ PrefixSumArgs prefix_sum_args;
    if (threadIdx.x == 0) {
        prefix_sum_args.partition_attr = args.materialized->l_partkey;
        prefix_sum_args.data_length = *args.materialized_size;
        prefix_sum_args.canonical_chunk_length = args.canonical_chunk_length;
        prefix_sum_args.padding_length = args.padding_length;
        prefix_sum_args.radix_bits = args.radix_bits;
        prefix_sum_args.ignore_bits = args.ignore_bits;
        prefix_sum_args.prefix_scan_state = args.prefix_scan_state;
        prefix_sum_args.tmp_partition_offsets = args.tmp_partition_offsets;
        prefix_sum_args.partition_offsets = args.partition_offsets;
    }*/
    

	//extern __shared__ uint8_t shared_mem[];
	__shared__ uint8_t prefix_sum_args_mem[sizeof(PrefixSumArgs)];
	const PrefixSumArgs& prefix_sum_args = *reinterpret_cast<PrefixSumArgs*>(prefix_sum_args_mem);
    if (threadIdx.x == 0) {
		new (reinterpret_cast<PrefixSumArgs*>(prefix_sum_args_mem)) PrefixSumArgs {
			args.materialized.l_partkey,
			*args.materialized_size,
			args.canonical_chunk_length,
			args.padding_length,
			args.radix_bits,
			args.ignore_bits,
			args.prefix_scan_state,
			args.tmp_partition_offsets,
			args.partition_offsets
		};
	}

    __syncthreads(); // FIXME use gridsync
    gpu_contiguous_prefix_sum_int32<<<gridDim.x, blockDim.x>>>(prefix_sum_args);
}
#endif
