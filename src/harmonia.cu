#include "harmonia.cuh"

#include <cub/util_debug.cuh>

namespace harmonia {

#ifdef NRDC
#error This file should not be compiled when rdc is disabled
#endif

// only contains the upper tree levels; stored in constant memory
// retain some space for kernel launch arguments (those are also stored in constant memory)
__constant__ child_ref_t harmonia_upper_levels[harmonia_max_constant_mem/sizeof(child_ref_t)];

}
