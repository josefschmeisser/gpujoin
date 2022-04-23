#include <numa-gpu/sql-ops/include/gpu_radix_partition.h>

// Combine the following to file into this single compilation unite in order to avoid using Relocatable Device Code (-rdc=true).
#include <numa-gpu/sql-ops/cudautils/gpu_common.cu>
#include <numa-gpu/sql-ops/cudautils/radix_partition.cu>
