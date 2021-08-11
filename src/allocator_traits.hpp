#pragma once

template<class T>
struct is_cuda_allocator {
    static constexpr bool value = false;
};

template<class T>
struct is_allocation_host_accessible {
    static constexpr bool value = true;
};
