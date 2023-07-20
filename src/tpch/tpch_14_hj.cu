#include "tpch_14_hj.cuh"

#include "common.hpp"
#include "cuda_utils.cuh"
#include "tpch_14_common.cuh"

__global__ void hj_build_kernel(const hj_args args) {
    const auto* __restrict__ p_partkey_column = args.part->p_partkey;

    //auto& ht = args.state->ht;
    const auto ht = args.ht;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < args.part_size; i += stride) {
        //ht.insert(p_partkey_column[i], i);
        hj_ht_t::insert(ht, p_partkey_column[i], i);
    }
}

__global__ void hj_probe_kernel(const hj_args args) {
    const char* prefix = "PROMO";

    const auto* __restrict__ l_shipdate_column = args.lineitem->l_shipdate;
    const auto* __restrict__ l_extendedprice_column = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount_column = args.lineitem->l_discount;
    const auto* __restrict__ l_partkey_column = args.lineitem->l_partkey;
    const auto* __restrict__ p_type_column = args.part->p_type;

    //auto& ht = args.state->ht;
    const auto ht = args.ht;

    numeric_raw_t numerator = 0;
    numeric_raw_t denominator = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.lineitem_size; i += stride) {
        if (l_shipdate_column[i] < lower_shipdate ||
            l_shipdate_column[i] >= upper_shipdate) {
            continue;
        }

        size_t part_tid;
        //bool match = ht.lookup(l_partkey_column[i], part_tid);
        bool match = hj_ht_t::lookup(ht, l_partkey_column[i], part_tid);
        // TODO use lane refill
        if (match) {
            const auto extendedprice = l_extendedprice_column[i];
            const auto discount = l_discount_column[i];
            const auto summand = extendedprice * (100 - discount);
            denominator += summand;

            const char* type = reinterpret_cast<const char*>(&p_type_column[part_tid]); // FIXME relies on undefined behavior
            if (device_strcmp(type, prefix, 5) == 0) {
                numerator += summand;
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
