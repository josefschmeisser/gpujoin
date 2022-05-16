#include "harmonia.cuh"

#if 0
namespace harmonia {

// only contains the upper tree levels; stored in constant memory
// retain some space for kernel launch arguments (those are also stored in constant memory)
//__constant__ uint32_t harmonia_upper_levels[14336];
__constant__ uint32_t harmonia_upper_levels[42*1024/sizeof(uint32_t)];

}
#endif
