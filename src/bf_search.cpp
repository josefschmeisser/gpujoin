#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>
#include <cassert>

/*
unsigned naive_lower_bound(Node* node, key_t key) {
    unsigned lower = 0;
    unsigned upper = node->count;
    do {
        unsigned mid = ((upper - lower) / 2) + lower;
        if (key < node->keys[mid]) {
            upper = mid;
        } else if (key > node->keys[mid]) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}
*/

unsigned branch_free_lower_bound_8(const int* haystack, unsigned size, int needle)
{
    size_t ret = (haystack[4] < needle) ? 4 : 0;
    ret += (haystack[ret + 2] < needle) ? 2 : 0;
    ret += (haystack[ret + 1] < needle) ? 1 : 0;
    ret += (haystack[ret] < needle) ? 1 : 0;
    return ret;
}

template<class T>
size_t branch_free_lower_bound(T needle, const T* arr, size_t size) {
    unsigned steps = 31 - __builtin_clz(size - 1);
    unsigned mid = 1 << steps;

    unsigned ret = (arr[mid] < needle) * (size - mid);
    while (mid > 0) {
        mid >>= 1;
        ret += (arr[ret + mid] < needle) ? mid : 0;
    }
    ret += (arr[ret] < needle) ? 1 : 0;

    return ret;
}

template<class T, size_t max_value>
size_t branch_free_lower_bound_unrolled(T needle, const T* arr, size_t size) {
    static constexpr auto steps = 31 - __builtin_clz(max_value - 1);

    unsigned mid = 1 << steps;
    unsigned ret = (arr[mid] < needle) * (size - mid);
    #pragma GCC unroll 8
    for (unsigned step = 1; step <= steps; ++step) {
        mid >>= 1;
        ret += (arr[ret + mid] < needle) ? mid : 0;
    }
    ret += (arr[ret] < needle) ? 1 : 0;

    return ret;
}

template<class T, unsigned max_step = 8>
size_t branch_free_exponential_search(T x, const T* arr, size_t n, float hint) {
    printf("=== search for: %d ===\n", x);
    unsigned lower = 0, upper = n;
    int start = static_cast<size_t>((n - 1)*hint);

    auto value = arr[start];

    if (value == x) {
        printf("=== match ===\n");
    }

    bool less = value < x;
    int offset = -1 + 2*less;// -2*less + 1;
    printf("less: %d offset: %d\n", less, offset);
    //start = (arr[start + offset] > x) ? start + offset : start;
    unsigned current = std::max(0, std::min<int>(n - 1 , start + offset));

    // less: (search to the right)
    // (2 < 3) == 1 -> t
    // (3 < 3) == 1 -> f
    // greater: (search to the left)
    // (3 < 2) == 0 -> t
    // (2 < 2) == 0 -> t
    // (1 < 2) == 0 -> f
    bool cont = ((arr[current] < x) == less);
    offset = cont ? offset<<1 : offset;
    current = std::max(0, std::min<int>(n - 1 , start + offset));
    printf("1. current: %u  arr[current]: %d offset: %d\n", current, arr[current], offset);

    cont = ((arr[current] < x) == less);
    offset = cont ? offset<<1 : offset;
    current = std::max(0, std::min<int>(n - 1 , start + offset));
    printf("2. current: %u  arr[current]: %d offset: %d\n", current, arr[current], offset);

    cont = ((arr[current] < x) == less);
    offset = cont ? offset<<1 : offset;
    current = std::max(0, std::min<int>(n - 1 , start + offset));
    printf("3. current: %u  arr[current]: %d offset: %d\n", current, arr[current], offset);

/*
    lower = start - (offset>>less);
    upper = start + (offset>>(1 - less));*/
    if (cont) {
        printf("+++ open bounds +++\n");
        //lower = less ? start + (offset>>less) : 0;
        //upper = less ? upper : start - (offset>>(1 - less));
        lower = less  ? std::max<int>(0, start + (offset>>less)) : lower;
        upper = !less ? std::min<int>(n, 1 + start + (offset>>(1 - less))) : upper;
    } else {
        printf("--- closed bounds ---\n");
        lower = std::max<int>(0, start + (offset>>less));
        upper = std::min<int>(n, 1 + start + (offset>>(1 - less)));
    }
/*
    auto pos = branch_free_lower_bound(x, arr, n);
    printf("lower: %lu upper: %lu position: %lu\n", lower, upper, pos);
*/
    return lower + branch_free_lower_bound(x, arr + lower, upper - lower);
}

template<class T, unsigned max_step = 8>
size_t branch_free_exponential_search2(T x, const T* arr, size_t n, float hint) {
    printf("=== search for: %d ===\n", x);
    unsigned lower = 0, upper = n;
    int start = static_cast<size_t>((n - 1)*hint);

    auto value = arr[start];

    if (value == x) {
        printf("=== match ===\n");
    }

    bool cont = true;
    bool less = value < x;
    int offset = -1 + 2*less;
    printf("less: %d offset: %d\n", less, offset);
    unsigned current = std::max(0, std::min<int>(n - 1 , start + offset));

    for (unsigned i = 0; i < max_step; ++i) {
        cont = ((arr[current] < x) == less);
        offset = cont ? offset<<1 : offset;
        current = std::max(0, std::min<int>(n - 1 , start + offset));
        printf("%u. current: %u  arr[current]: %d offset: %d\n", i, current, arr[current], offset);
    }
    printf("lower: %u upper: %u\n", lower, upper);

    const auto pre_lower = std::max<int>(0, std::min<int>(n, start + (offset>>less)));
    const auto pre_upper = 1 + std::max<int>(0, std::min<int>(n, start + (offset>>(1 - less))));

    printf("pre_lower: %d pre_upper: %d\n", pre_lower, pre_upper);
    if (cont) {
        printf("+++ open bounds +++\n");
    }

    lower = (!cont || less) ? pre_lower : lower;
    upper = (!cont || !less) ? pre_upper : upper;

    printf("lower: %d upper: %d\n", lower, upper);

    return lower + branch_free_lower_bound(x, arr + lower, upper - lower);
}

int main() {
    const std::vector<int> data = { 1, 2, 4, 5, 5, 6 , 8, 12, 20, 20, 21, 23};

    for (int i = 0; i <= data.back(); ++i) {
        const auto it = std::lower_bound(data.begin(), data.end(), i);
        const auto std_pos = it - data.begin();
        /*
        auto pos = branch_free_lower_bound(i, data.data(), data.size());
        auto pos2 = branch_free_lower_bound_unrolled<int, 12>(i, data.data(), data.size());
        std::cout << "value: " << i <<  " std::lower_bound: " << it - data.begin() << " bf: " << pos << " bf unrolled: " << pos2 << std::endl;
        */

        //auto pos = branch_free_exponential_search(i, data.data(), data.size(), 0.5f);
        auto pos = branch_free_exponential_search2<int, 4>(i, data.data(), data.size(), 0.5f);
        assert(std_pos == pos);
        printf("std::lower_bound: %lu pos: %lu\n", std_pos, pos);
    }
    return 0;
}
