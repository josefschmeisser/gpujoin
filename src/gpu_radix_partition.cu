//#include <numa-gpu/sql-ops/include/gpu_radix_partition.h>
#include "gpu_radix_partition.cuh"

// Combine the following to file into this single compilation unite in order to avoid using Relocatable Device Code (-rdc=true).
#include <numa-gpu/sql-ops/cudautils/gpu_common.cu>
#include <numa-gpu/sql-ops/cudautils/radix_partition.cu>

__global__ void partitioned_consumer_assign_tasks(partitioned_consumer_assign_tasks_args args) {
    const auto fanout = 1U << args.radix_bits;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t rel_size = args.rel_length - args.rel_padding_length*fanout;
        const uint32_t avg_task_size = (rel_size + gridDim.x - 1U) / gridDim.x;

        args.task_assignment[0] = 0U;
        uint32_t task_id = 1U;
        uint32_t task_size = 0U;
        for (uint32_t p = 0U; p < fanout && task_id < gridDim.x; ++p) {
            const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
            const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

            task_size += partition_size;
            if (task_size >= avg_task_size) {
                args.task_assignment[task_id] = p + 1U;
#ifdef DEBUG
        printf("Assigning partitions [%u, %u] to block %d\n",
               args.task_assignments[task_id - 1],
               args.task_assignments[task_id], task_id);
#endif

                task_size = 0U;
                task_id += 1;
            }
        }

        for (uint32_t i = task_id; i <= gridDim.x; ++i) {
            args.task_assignment[i] = fanout;
        }
    }
}
