#include <cstddef>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <cerrno>
#include <cstring>

#include <sys/mman.h>
#include <linux/mman.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>

size_t round_to_next_page(size_t size, size_t page_size) {
    const auto align_mask = ~(page_size - 1);
    return (size + page_size - 1) & align_mask;
}

int main() {
    size_t len = 8ul * 1024ul*1024ul;
    void *ptr = mmap(NULL, len, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,// MAP_HUGE_1GB,
                 -1, 0);
    if (!ptr) {
        throw std::runtime_error("mmap failed");
    }

    const auto aligned_size = round_to_next_page(len, 1 << 21);
    unsigned long node_mask = 1<<0;
    unsigned long mymask = *numa_get_mems_allowed()->maskp;
    unsigned long maxnode = numa_get_mems_allowed()->size;

    printf("mymask: %p maxnode: %lu\n", mymask, maxnode);
    const auto r = mbind(ptr, aligned_size, MPOL_BIND, &mymask, maxnode, MPOL_MF_STRICT);

    if (r != 0) {
        std::cout << "mbind failed: " << std::strerror(errno) << '\n';
        throw std::runtime_error("mbind failed");
    }

    std::cout << "writting to mem..." << std::endl;
    std::getc(stdin);

    return 0;
}

void printmask(char *name, struct bitmask *mask) {
    printf("%s: ", name);
    for (int i = 0; i < mask->size; i++) {
        if (numa_bitmask_isbitset(mask, i)) {
            printf("%d ", i);
        }
    }
    putchar('\n');
}

int main3() {
    cpu_set_t mask;
    long nproc, i;
    struct bitmask *numa_nodes = numa_allocate_nodemask();

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        exit(-1);
    }

    nproc = sysconf(_SC_NPROCESSORS_ONLN);
    for (i = 0; i < nproc; i++) {
        if (CPU_ISSET(i, &mask)) {
            printf("core %d (NUMA node %d)\n", i, numa_node_of_cpu(i));
            numa_bitmask_setbit(numa_nodes, numa_node_of_cpu(i));
        }
    }

    printmask("worker nodes", numa_nodes);
    return 0;
}
