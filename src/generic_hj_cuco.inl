#include "cuda_utils.cuh"
#include "index_lookup_config.hpp"

#include <sys/types.h>
#include <cuco/static_multimap.cuh>
#include <cooperative_groups.h>

template<uint32_t block_size, uint32_t cg_size, class ArgsType>
__global__ void hj_cuco_build_kernel(ArgsType args) {
    namespace cg = cooperative_groups;
    auto g = cg::tiled_partition<cg_size>(cg::this_thread_block());
    args.map_mutable_view.insert(g, cuco::pair{1, 1});
    // TODO
}

template<class ArgsType >
__global__ void hj_cuco_probe_kernel(const ArgsType args) {

}
