#include <iostream>

#include <sys/mman.h>

int main() {

    void *ptr = mmap(NULL, 8 * (1 << 21), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                 -1, 0);

    return 0;
}
