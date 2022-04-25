#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <thread>
#include <string>
#include <string_view>
#include <iostream>
#include <functional>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.hpp"
#include "scope_guard.hpp"

static constexpr bool serialize = false;
static constexpr size_t min_partition_size = 32*1024*1024;

unsigned sample_line_width(const char* data_start, size_t data_size);

ssize_t find_first(uint64_t pattern, const char* begin, size_t len);

int64_t to_int(std::string_view s);

template<typename T>
struct input_parser;

template<>
struct input_parser<int> {
    static bool parse(const char* begin, size_t len, int& result) {
        // TODO parse sign
        bool invalid = (len < 1);
        result = 0;
        for (size_t i = 0; i < len; ++i) {
            char c = begin[i];
            invalid |= (c < '0' || c > '9');
            result = result * 10 + (begin[i] - '0');
        }
        return !invalid;
    }
};

template<>
struct input_parser<uint32_t> {
    static bool parse(const char* begin, size_t len, uint32_t& result) {
        bool invalid = (len < 1);
        result = 0;
        for (size_t i = 0; i < len; ++i) {
            char c = begin[i];
            invalid |= (c < '0' || c > '9');
            result = result * 10 + (begin[i] - '0');
        }
        return !invalid;
    }
};

template<>
struct input_parser<int64_t> {
    static bool parse(const char* begin, size_t len, int64_t& result) {
        // TODO parse sign
        bool invalid = (len < 1);
        result = 0;
        for (size_t i = 0; i < len; ++i) {
            char c = begin[i];
            invalid |= (c < '0' || c > '9');
            result = result * 10 + (begin[i] - '0');
        }
        return !invalid;
    }
};

template<>
struct input_parser<uint64_t> {
    static bool parse(const char* begin, size_t len, uint64_t& result) {
        bool invalid = (len < 1);
        result = 0;
        for (size_t i = 0; i < len; ++i) {
            char c = begin[i];
            invalid |= (c < '0' || c > '9');
            result = result * 10 + (begin[i] - '0');
        }
        return !invalid;
    }
};

template<unsigned Precision, unsigned Scale>
struct input_parser<numeric<Precision, Scale>> {
    static bool parse(const char* begin, size_t len, numeric<Precision, Scale>& result) {
        constexpr uint64_t period_pattern = 0x2E2E2E2E2E2E2E2Eull;
        std::string_view numeric_view(begin, len);
        ssize_t dot_position = find_first(period_pattern, begin, len);
        int64_t& numeric_raw = result.raw;

        if (dot_position < 0) {
            // no dot
            numeric_raw = 100*to_int(numeric_view.substr(0, len));
        } else if (dot_position == 0) {
            auto part2 = numeric_view.substr(dot_position + 1); // TODO limit number of digits
            numeric_raw = to_int(part2);
        } else {
            auto part1 = numeric_view.substr(0, dot_position);
            auto part2 = numeric_view.substr(dot_position + 1);
            numeric_raw = to_int(part1) * 100 + to_int(part2); // TODO scale and limit number of digits
        }
        return true;
    }
};

template<>
struct input_parser<date> {
    // source: https://stason.org/TULARC/society/calendars/2-15-1-Is-there-a-formula-for-calculating-the-Julian-day-nu.html
    static constexpr uint32_t to_julian_day(uint32_t day, uint32_t month, uint32_t year) {
        uint32_t a = (14 - month) / 12;
        uint32_t y = year + 4800 - a;
        uint32_t m = month + 12 * a - 3;
        return day + (153 * m + 2) / 5 + y * 365 + y / 4 - y / 100 + y / 400 - 32045;
    }

    static constexpr void from_julian_day(uint32_t julian_day, uint32_t& year, uint32_t& month, uint32_t& day) {
        uint32_t a = julian_day + 32044;
        uint32_t b = (4*a+3)/146097;
        uint32_t c = a-((146097*b)/4);
        uint32_t d = (4*c+3)/1461;
        uint32_t e = c-((1461*d)/4);
        uint32_t m = (5*e+2)/153;

        day = e - ((153*m+2)/5) + 1;
        month = m + 3 - (12*(m/10));
        year = (100*b) + d - 4800 + (m/10);
    }

    static uint32_t parse(const char* begin, size_t len, date& result) {
        if (len != 10) return false;

        bool valid = true;
        uint32_t day = 0, month = 0, year = 0;
        for (unsigned i = 0; i < 4; ++i) {
            char c = begin[i];
            valid &= (c >= '0' && c <= '9');
            year = year * 10 + (begin[i] - '0');
        }
        valid &= begin[4] == '-';
        // the month is expected to be zero-padded
        for (unsigned i = 5; i < 7; ++i) {
            char c = begin[i];
            valid &= (c >= '0' && c <= '9');
            month = month * 10 + (begin[i] - '0');
        }
        valid &= begin[7] == '-';
        // the day is expected to be zero-padded
        for (unsigned i = 8; i < 10; ++i) {
            char c = begin[i];
            valid &= (c >= '0' && c <= '9');
            day = day * 10 + (begin[i] - '0');
        }
        result.raw = to_julian_day(day, month, year);

        return valid;
    }
};

template<>
struct input_parser<char> {
    static bool parse(const char* begin, size_t len, char& result) {
        result = begin[0];
        return (len == 1);
    }
};

template<size_t N>
struct input_parser<std::array<char, N>> {
    static bool parse(const char* begin, size_t len, std::array<char, N>& result) {
        std::memcpy(result.data(), begin, std::min(N, len));
        return (N >= len);
    }
};


template<template<typename T> class Mapper, typename... Ts>
struct map_tuple;

template<template<typename T> class Mapper, typename... Ts>
struct map_tuple<Mapper, std::tuple<Ts...>> {
    using mapped_type = std::tuple<typename Mapper<Ts>::type...>;
};

template<template<typename U> typename VectorAllocator, typename T>
struct to_unique_ptr_to_vector {
    template<typename V> using partial = to_unique_ptr_to_vector<VectorAllocator, V>;
    using type = std::unique_ptr<std::vector<T, VectorAllocator<T>>>;
};


template<class F, class Tuple, std::size_t... I>
constexpr decltype(auto) tuple_foreach_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return (std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))), ...);
}

template<class F, class Tuple>
constexpr decltype(auto) tuple_foreach(F&& f, Tuple&& t) {
    return tuple_foreach_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}


template<typename T, unsigned I, bool L>
struct tuple_entry_descriptor {
    using type = T;
    static constexpr unsigned index = I;
    static constexpr bool is_last = L;
};

template<class Tuple>
struct tuple_foreach_type {
    template<class F, std::size_t... I>
    static constexpr decltype(auto) invoke_impl(F&& f, std::index_sequence<I...>) {
        return (std::invoke(std::forward<F>(f), tuple_entry_descriptor<typename std::tuple_element<I, Tuple>::type, I, I+1 == sizeof...(I)>()), ...);
    }

    template<class F>
    static constexpr decltype(auto) invoke(F&& f) {
        return invoke_impl(
            std::forward<F>(f),
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
};


template<class Tuple>
struct tuple_for_index {
    template<class F, std::size_t... I>
    static constexpr decltype(auto) invoke_impl(F&& f, std::index_sequence<I...>) {
        return (std::invoke(std::forward<F>(f), I), ...);
    }

    template<class F>
    static constexpr decltype(auto) invoke(F&& f) {
        return invoke_impl(
            std::forward<F>(f),
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
};


template<typename TupleType, template<typename T> typename VectorAllocator>
struct worker {
    using dest_tuple_type = typename map_tuple<to_unique_ptr_to_vector<VectorAllocator, void>::template partial, TupleType>::mapped_type;

    const char* data_start_;
    const char* partition_start_hint_;
    const size_t partition_size_hint_;
    const unsigned thread_count_;
    const unsigned thread_num_;

    const char* partition_start_;
    size_t partition_size_;

    size_t last_index_;
    size_t count_ = 0;

    worker(const char* data_start, const char* partition_start_hint, size_t partition_size_hint, unsigned thread_count, unsigned thread_num)
        : data_start_(data_start)
        , partition_start_hint_(partition_start_hint)
        , partition_size_hint_(partition_size_hint)
        , thread_count_(thread_count)
        , thread_num_(thread_num)
    {}

    void initial_run(dest_tuple_type& dest, size_t dest_begin);

    void run(dest_tuple_type& dest, size_t dest_begin);
};

template<typename TupleType, template<typename T> typename VectorAllocator>
void worker<TupleType, VectorAllocator>::initial_run(dest_tuple_type& dest, size_t dest_begin) {
    //printf("=== in worker #%u ===\n", thread_num_);
/*
    printf("worker #%u initial_run partition_start_hint_: %p\n", thread_num_, partition_start_hint_);
    printf("worker #%u hinted first line: %.*s\n", thread_num_, 120, partition_start_hint_);
*/

    // correct partition size
    if (thread_num_ > 0) {
        size_t offset = 0;
        const size_t max_offset = partition_start_hint_ - data_start_;
        while (offset < partition_size_hint_) {
            if (offset > max_offset) {
                // partition to small
                assert(false);
            } else if (*(partition_start_hint_ - offset) == '\n') {
                break;
            } else {
                offset++;
            }
        }
        --offset; // skip over the newline character
        partition_start_ = partition_start_hint_ - offset;
        partition_size_ = partition_size_hint_ + offset;
    } else {
        // first partition
        partition_start_ = partition_start_hint_;
        partition_size_ = partition_size_hint_;
    }

    //printf("worker #%u initial_run partition_start_: %p\n", thread_num_, partition_start_);
    run(dest, dest_begin);
}

std::ostream& operator<<(std::ostream& os, const date& value);

template<size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<char, N>& arr) {
    const auto len = strnlen(arr.data(), N);
    os << std::string_view(arr.data(), len);
    return os;
}

template<unsigned Precision, unsigned Scale>
std::ostream& operator<<(std::ostream& os, const numeric<Precision, Scale>& value) {
    auto r = std::div(value.raw, 100);
    os << r.quot;
    if (r.rem != 0) {
        os << "." << r.rem;
    }
    return os;
}

template<typename TupleType, template<typename T> typename VectorAllocator>
void worker<TupleType, VectorAllocator>::run(worker<TupleType, VectorAllocator>::dest_tuple_type& dest, size_t dest_begin) {
    constexpr uint64_t bar_pattern = 0x7C7C7C7C7C7C7C7Cull;
    constexpr uint64_t newline_pattern = 0x0A0A0A0A0A0A0A0Aull;
    constexpr auto column_count = std::tuple_size<TupleType>::value;

    const bool is_last_partition = (thread_num_ + 1 == thread_count_);
    const auto dest_limit = std::get<0>(dest)->size();
    size_t dest_index = dest_begin + thread_num_;
    size_t i = 0;
    //printf("worker #%u first line: %.*s\n", thread_num_, 120, partition_start_);
    while (i < partition_size_ && dest_index < dest_limit) {
        //std::cout << "line: " << count_ << " ";
        const auto line_start = i;
        ssize_t sep_pos;//, newline_pos;
        unsigned sep_cnt = 0;
        bool line_valid = true;
        tuple_foreach_type<TupleType>::invoke([&](auto column_desc_inst) {
            using column_desc_type = decltype(column_desc_inst);
            using element_type = typename decltype(column_desc_inst)::type;
            constexpr auto index = column_desc_type::index;

            const auto remaining = partition_size_ - i;
            if constexpr (column_desc_type::is_last) {
                sep_pos = find_first(newline_pattern, partition_start_ + i, remaining);
                sep_pos = (is_last_partition && sep_pos < 0) ? remaining : sep_pos; // treat the end of file as a regular separator
            } else {
                sep_pos = find_first(bar_pattern, partition_start_ + i, remaining);
            }
            sep_cnt += (sep_pos >= 0);
            sep_pos = (sep_pos < 0) ? remaining : sep_pos;
            //printf("\nvalue: %.*s\n", sep_pos, partition_start_ + i);

            auto& value = (*std::get<index>(dest))[dest_index];
            line_valid &= input_parser<element_type>::parse(partition_start_ + i, sep_pos, value);

            //std::cout << "|" << value;

            i += sep_pos + 1;
        });

        //std::cout << std::endl;

        line_valid &= sep_cnt == column_count;
        if (!line_valid && i >= partition_size_) {
            // incomplete line at the end of this partition
            /*
            long remaining = static_cast<long>(partition_size_) - i;
            printf("worker #%u partition exhausted - remaining: %ld\n", thread_num_, remaining);*/
            break;
        } else if (!line_valid) {
            // invalid line somewhere in the partion
            std::cerr << "invalid line at byte " << line_start << std::endl;/*
            long remaining = static_cast<long>(partition_size_) - i;
            printf("worker #%u invalid line - remaining: %ld sep_pos: %ld\n", thread_num_, remaining, sep_pos);
            size_t to_print = partition_size_ - line_start;
            printf("worker #%u linep: %.*s\n", thread_num_, to_print, partition_start_ + line_start);
            printf("worker #%u liner: %.*s\n", thread_num_, 130, partition_start_ + line_start);*/
            return;
        }

        last_index_ = dest_index;
        dest_index += thread_count_;

        ++count_;
    }
}

unsigned sample_line_width(const char* data_start, size_t data_size);

template<class TupleType, template<typename T> typename VectorAllocator>
void densify(std::vector<worker<TupleType, VectorAllocator>>& workers, typename worker<TupleType, VectorAllocator>::dest_tuple_type& dest) {
    size_t count = 0;
    std::vector<unsigned> state; // worker ids
    for (auto& worker : workers) {
        count += worker.count_;
        state.emplace_back(worker.thread_num_);
    }
    std::sort(state.begin(), state.end(), [&workers](const auto& a, const auto& b) {
        return workers[a].last_index_ < workers[b].last_index_;
    });

    //printf("start densifying...\n");
    const auto num_workers = workers.size();

    size_t dense_upper_limit = workers[0].last_index_; // the vectors are dense up to this index
    for (unsigned i = 1; i < num_workers; ++i) {
        if (workers[i].last_index_ < dense_upper_limit) {
            dense_upper_limit = workers[i].last_index_;
        }
    }

    auto occupied = [&workers](size_t pos, unsigned worker_in_charge_id) {
        return workers[worker_in_charge_id].last_index_ >= pos;
    };

    // densify
    unsigned original_worker_id = dense_upper_limit % num_workers;
    size_t i;
    for (i = dense_upper_limit + 1;; ++i) {
        ++original_worker_id;
        original_worker_id = (original_worker_id >= num_workers) ? 0 : original_worker_id;
        //assert(i % num_workers == original_worker_id);

        unsigned last_id;
        size_t* last_index;

        while (!state.empty()) {
            last_id = state.back();
            last_index = &workers[last_id].last_index_;
            if (*last_index <= i) {
                state.pop_back();
            } else {
                break;
            }
        }
        if (state.empty()) break;

        if (occupied(i, original_worker_id)) continue;

        assert(i < *last_index);

        // move elements from last_index to i
        tuple_foreach_type<TupleType>::invoke([&](auto column_desc_inst) {
            using column_desc_type = decltype(column_desc_inst);
            constexpr auto index = column_desc_type::index;

            auto& vec = (*std::get<index>(dest));
            vec[i] = vec[*last_index];
        });

        *last_index -= num_workers;
    }

    // truncate vectors
    tuple_foreach([&](auto& element) {
        element->resize(count);
    }, dest);
}


template<typename... Ts>
void allocate_vectors(std::tuple<Ts...>& input_tuple, size_t n) {
    std::apply([&n](Ts&... Args) {
        ((Args.reset(new typename Ts::element_type()), Args->resize(n)), ...); // the unique_ptrs will be allocated with the default allocator
    }, input_tuple);
}

template<typename TupleType, template<typename T> typename VectorAllocator>
auto create_vectors(size_t n) {
    using mapped_type = typename map_tuple<to_unique_ptr_to_vector<VectorAllocator, void>::template partial, TupleType>::mapped_type;
    mapped_type new_tuple;
    allocate_vectors(new_tuple, n);
    return new_tuple;
}


template<typename TupleType, template<typename T> typename VectorAllocator = std::allocator>
auto parse(const std::string& file) {
    // open file handle
    auto file_guard = make_scope_guard(open(file.c_str(), O_RDONLY), -1, &close);
    int handle = file_guard.get(); // TODO
    if (handle == -1) {
        throw std::runtime_error(std::string("`open` failed: ") + std::strerror(errno));
    }

    // determine file length
    lseek(handle, 0, SEEK_END);
    auto size = lseek(handle, 0, SEEK_CUR);
    assert(size > 0);

    // ensure that the mapping size is a multiple of 8 (bytes beyound the file's
    // region are set to zero)
    auto mapping_size = size + 8; // padding for the last partition
    // https://stackoverflow.com/questions/47604431/why-we-can-mmap-to-a-file-but-exceed-the-file-size
    
    auto mmap_guard = make_scope_guard(mmap(nullptr, mapping_size, PROT_READ, MAP_SHARED, handle, 0), static_cast<void*>(nullptr), [&mapping_size](void* ptr) {
        munmap(ptr, mapping_size);
    });
    void* data = mmap_guard.get();
    if (data == nullptr) {
        throw std::runtime_error(std::string("`mmap` failed: ") + std::strerror(errno));
    }
    const char* input = reinterpret_cast<const char*>(data);

    const auto est_line_width = sample_line_width(input, size);
    //std::cout << "estimated line width: " << est_line_width << std::endl;
    const auto est_record_count = size/est_line_width;
    //std::cout << "estimated line count: " << est_record_count << std::endl;

    const auto num_threads = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), size/min_partition_size));

    std::vector<int32_t> dest1;
    dest1.resize(est_record_count);
    auto dest_tuple = create_vectors<TupleType, VectorAllocator>(est_record_count);
    std::vector<worker<TupleType, VectorAllocator>> workers;

    std::thread threads[num_threads];

    size_t remaining = size;
    size_t partition_size = size / num_threads;
    const char* data_start = static_cast<const char*>(data);
    const char* partition_start_hint = data_start;

    //printf("estimated partition size: %lu\n", partition_size);

    // create workers
    for (unsigned i = 0; i < num_threads; ++i) {
        size_t size_hint = std::min(remaining, partition_size);
        remaining -= size_hint;
        //printf("partition #%u partition_start_hint: %p size_hint: %lu\n", i, partition_start_hint, size_hint);
        // worker(const char* data_start, const char* partition_start_hint, size_t partition_size_hint, unsigned thread_num)
        workers.emplace_back(data_start, partition_start_hint, size_hint, num_threads, i);
        partition_start_hint += size_hint;
    }
    // launch workers
    for (unsigned i = 0; i < num_threads; ++i) {
        threads[i] = std::thread([&](unsigned thread_num) {
            auto& my_worker = workers[thread_num];
            my_worker.initial_run(std::ref(dest_tuple), 0ull);
        }, i);
        if constexpr (serialize) {
            threads[i].join();
        }
    }
    if constexpr (!serialize) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads[i].join();
        }
    }

    // fill potential gaps in the result vectors
    densify(workers, dest_tuple);

    return dest_tuple;
}


template<class Tuple, std::size_t... I>
void do_sort_impl(Tuple&& tuple, std::index_sequence<I...>) {
    auto& key_vec = *std::get<0>(tuple);
    auto permutation = compute_permutation(key_vec.begin(), key_vec.end(), std::less<>{});
    apply_permutation(permutation, (*std::get<I>(tuple))...);
}

template<class Tuple>
void do_sort(Tuple&& t) {
    do_sort_impl(
        std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

template<class Tuple>
void write_out(Tuple&& t, std::ostream& os) {
    auto& vec0 = *std::get<0>(t);

    for (size_t i = 0; i < vec0.size(); ++i) {
        bool first = true;
        tuple_foreach([&](auto& vec) {
            if (!first) {
                os << "|";
            }
            os << (*vec)[i];
            first = false;
        }, t);
        os << "\n";
    }
}
