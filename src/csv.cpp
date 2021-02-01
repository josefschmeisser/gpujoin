//#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <tuple>
#include <vector>
#include <thread>
#include <string>
#include <string_view>
#include <iostream>
#include <functional>

#include <bits/stdint-uintn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

static constexpr bool serialize = true;

int64_t to_int(std::string_view s) {
    int64_t result = 0;
    for (auto c : s) result = result * 10 + (c - '0');
    return result;
}


/**
 * @brief Get all matches in the given character block
 * 
 * @param pattern the character to search broadcasted into a 64-bit integer
 * @param block memory block in which to search
 * @return uint64_t 64-bit integer with all the matches
 */
inline uint64_t get_matches(uint64_t pattern, uint64_t block) {
    constexpr uint64_t high = 0x8080808080808080ull;
    constexpr uint64_t low = 0x7F7F7F7F7F7F7F7Full;
    uint64_t lowChars = (~block) & high;
    uint64_t foundChars = ~((((block & low) ^ pattern) + low) & high);
    uint64_t matches = foundChars & lowChars;
    return matches;
}

// pattern : the character to search broadcasted into a 64bit integer
// begin : points somewhere into the partition
// len : remaining length of the partition
// return : the position of the first matching character or -1 otherwise
ssize_t find_first(uint64_t pattern, const char* begin, size_t len) {
    //printf("find_first begin: %p length: %lu\n", begin, len);
    assert(len > 0);
    // we may assume that reads from 'begin' within [len, len + 8) yield zero
    for (size_t i = 0; i < len; i += 8) {
        uint64_t block = *reinterpret_cast<const uint64_t*>(begin + i);
        uint64_t matches = get_matches(pattern, block);
        if (matches != 0) {
            uint64_t pos = __builtin_ctzll(matches) / 8;
            if (pos < 8) {
                const auto real_pos = i + pos;
                return (real_pos >= len) ? -1 : real_pos;
            }
        }
    }
    return -1;
}

int32_t read_int(const char* begin, size_t len) {
    bool invalid = (len < 1);
    int32_t result = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = begin[i];
        invalid |= (c >= '0' && c <= '9');
        result = result * 10 + (begin[i] - '0');
    }
    return result;
}

#if 0
// begin : points somewhere into the partition
// len : remaining length of the partition
// return : 64-bit integer representation of the provided numeric
int64_t read_numeric(const char* begin, size_t len) {
    constexpr uint64_t period_pattern = 0x2E2E2E2E2E2E2E2Eull;
    std::string_view numeric_view(begin, len);
    ssize_t dot_position = find_first(period_pattern, begin, len);
    if (dot_position < 1) {
        std::cerr << "invalid numeric literal" << std::endl;
        return 0;
    }
    auto part1 = numeric_view.substr(0, dot_position);
    auto part2 = numeric_view.substr(dot_position + 1);
    int64_t numeric = to_int(part1) * 100 + to_int(part2); // TODO
    return numeric;
}
#endif

template<unsigned Precision, unsigned Scale>
struct numeric {
    static constexpr auto precision = Precision;
    static constexpr auto scale = Scale;
    uint64_t raw;
};

struct date {
    uint32_t raw;
};

template<typename T>
struct input_parser;
/*
struct lineitem_table_t {
    std::vector<uint32_t> l_orderkey;
    std::vector<uint32_t> l_partkey;
    std::vector<uint32_t> l_suppkey;
    std::vector<uint32_t> l_linenumber;
    std::vector<int64_t> l_quantity;
    std::vector<int64_t> l_extendedprice;
    std::vector<int64_t> l_discount;
    std::vector<int64_t> l_tax;
    std::vector<char> l_returnflag;
    std::vector<char> l_linestatus;
    std::vector<uint32_t> l_shipdate;
    std::vector<uint32_t> l_commitdate;
    std::vector<uint32_t> l_receiptdate;
    std::vector<std::array<char, 25>> l_shipinstruct;
    std::vector<std::array<char, 10>> l_shipmode;
    std::vector<std::array<char, 44>> l_comment;
};
*/
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

template<unsigned Precision, unsigned Scale>
struct input_parser<numeric<Precision, Scale>> {
    static bool parse(const char* begin, size_t len, numeric<Precision, Scale>& result) {
        /*
result.raw = 0;
return true;*/
//printf("parse numeric() len: %lu\n", len);
        constexpr uint64_t period_pattern = 0x2E2E2E2E2E2E2E2Eull;
        std::string_view numeric_view(begin, len);
        ssize_t dot_position = find_first(period_pattern, begin, len);
      //  std::cout << "dot_position: " << dot_position << std::endl;

/*
        if (dot_position < 1) {
            std::cerr << "invalid numeric literal" << std::endl;
            return false;
        }
        auto part1 = numeric_view.substr(0, dot_position);
        auto part2 = numeric_view.substr(dot_position + 1);
        int64_t numeric_raw = to_int(part1) * 100 + to_int(part2); // TODO
        result.raw = numeric_raw;
        return true;*/
        if (dot_position < 0) {
            // no dot
            int64_t numeric_raw = to_int(numeric_view.substr(0, len));
      //      cout << "\nparse numeric without dot:  " << numeric_raw << std::endl;
            result.raw = numeric_raw;
        } else if (dot_position == 0) {
            // TODO
            assert(false);
        } else {
            auto part1 = numeric_view.substr(0, dot_position);
            auto part2 = numeric_view.substr(dot_position + 1);
     //       cout << "\nparse numeric: " << part1 << " . " << part2 << std::endl;
            int64_t numeric_raw = to_int(part1) * 100 + to_int(part2); // TODO scale
            result.raw = numeric_raw;
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
    
    static uint32_t parse(const char* begin, size_t len, date& result) {
        if (len != 10) return false;
/*
        uint32_t day, month, year;
        int parsed = sscanf(begin, "%4d-%2d-%2d", &year, &month, &day);
        bool valid = parsed == 3;
        result.raw = to_julian_day(day, month, year);
        return valid;*/

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
        //std::cout << "\n" << year << "-" << month << "-" << day << " valid: " << valid << std::endl;

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
        return (N == len);
    }
};


template<template<typename T> class Mapper, typename... Ts>
struct map_tuple;

template<template<typename T> class Mapper, typename... Ts>
struct map_tuple<Mapper, std::tuple<Ts...>> {
    using mapped_type = std::tuple<typename Mapper<Ts>::type...>;
};

template<class T>
struct to_unique_ptr_to_vector {
    using type = std::unique_ptr<std::vector<T>>;
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



//template<typename ColumnTypes, bool first_partition = false>
template<typename TupleType>
struct worker {
    using dest_tuple_type = typename map_tuple<to_unique_ptr_to_vector, TupleType>::mapped_type;

    const char* data_start_;
    const char* partition_start_hint_;
    const size_t partition_size_hint_;
    const unsigned thread_count_;
    const unsigned thread_num_;

    const char* partition_start_;
    size_t partition_size_;

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

template<typename TupleType>
void worker<TupleType>::initial_run(dest_tuple_type& dest, size_t dest_begin) {
    printf("=== in worker #%u ===\n", thread_num_);
    //printf("initial_run partition_start_hint_: %p\n", partition_start_hint_);
    fflush(stdout);

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

    run(dest, dest_begin);
}

template<size_t N>
ostream& operator<<(ostream& os, const std::array<char, N>& arr) {
    os << std::string_view(arr.data(), N);
    return os;
}

ostream& operator<<(ostream& os, const date& value) {
    os << value.raw; // TODO
    return os;
}

template<unsigned Precision, unsigned Scale>
ostream& operator<<(ostream& os, const numeric<Precision, Scale>& value) {
    os << value.raw; // TODO
    return os;
}

template<typename TupleType>
void worker<TupleType>::run(worker<TupleType>::dest_tuple_type& dest, size_t dest_begin) {
    constexpr uint64_t bar_pattern = 0x7C7C7C7C7C7C7C7Cull;
    constexpr uint64_t newline_pattern = 0x0A0A0A0A0A0A0A0Aull;

    constexpr auto column_count = tuple_size<TupleType>::value;

    const auto dest_limit = std::get<0>(dest)->size();
    size_t dest_index = dest_begin + thread_num_;
    size_t i = 0;

    //std::cout << "dest_limit: " << dest_limit << std::endl;
    while (i < partition_size_ &&  dest_index < dest_limit) {
        const auto line_start = i;
        ssize_t sep_pos;
        unsigned sep_cnt = 0;

        tuple_foreach_type<TupleType>::invoke([&](auto column_desc_inst) {
            using column_desc_type = decltype(column_desc_inst);
            using element_type = typename decltype(column_desc_inst)::type;
            constexpr auto index = column_desc_type::index;

            if constexpr (column_desc_type::is_last) {
                sep_pos = find_first(newline_pattern, partition_start_ + i, partition_size_ - i);
            } else {
                sep_pos = find_first(bar_pattern, partition_start_ + i, partition_size_ - i);
            }
            sep_cnt += (sep_pos >= 0);
            sep_pos = std::max(sep_pos, 0l);

            auto& value = (*std::get<index>(dest))[dest_index];
            bool valid = input_parser<element_type>::parse(partition_start_ + i, sep_pos, value);
            // TODO

            std::cout << "|" << value;

            i += sep_pos + 1;
        });

        auto newline_pos = find_first(newline_pattern, partition_start_ + i, partition_size_ - i);
        i += newline_pos + 1;
        std::cout << std::endl;

        if (sep_cnt != column_count) {
            std::cerr << "invalid line at byte " << line_start << std::endl;
            return;
        }

        dest_index += thread_count_;

      //  if (i > 300) break;
    }
}








unsigned sample_line_width(const char* data_start, size_t data_size) {
    constexpr uint64_t newline_pattern = 0x0A0A0A0A0A0A0A0Aull;

    ssize_t newline_pos;
    size_t acc = 0, count = 0;
    for (size_t i = 0; i < data_size && count < 10; i += newline_pos + 1) {
        newline_pos = find_first(newline_pattern, data_start + i, data_size - i);

        if (i == 0) continue; // skip first line
        else if (newline_pos < 0) break;

        acc += newline_pos;
        ++count;
    }

    std::cout << "count: " << count << " acc: " << acc << std::endl;

    return (count > 0) ? acc/count : data_size;
}

void rebalance() {
}



template<typename... Ts>
void allocate_vectors(std::tuple<Ts...>& input_tuple, size_t n) {
    std::apply([&n](Ts&... Args) {
        ((Args.reset(new typename Ts::element_type()), Args->resize(n)), ...);
    }, input_tuple);
}

template<typename TupleType>
auto create_vectors(size_t n) {
    using mapped_type = typename map_tuple<to_unique_ptr_to_vector, TupleType>::mapped_type;
    mapped_type new_tuple;
    allocate_vectors(new_tuple, 10); // TODO
    return new_tuple;
}


template<typename TupleType>
void parse(const std::string& file) {

    int handle = open(file.c_str(), O_RDONLY);
    lseek(handle, 0, SEEK_END);
    auto size = lseek(handle, 0, SEEK_CUR);
    cout << "size: " << size << std::endl;
    // ensure that the mapping size is a multiple of 8 (bytes beyound the file's
    // region are set to zero)
    auto mapping_size = size + 8;  // padding for the last partition
    //auto data = mmap(nullptr, size, PROT_READ, MAP_SHARED, handle, 0);
    void* data = mmap(nullptr, mapping_size, PROT_READ, MAP_SHARED, handle, 0);
    const char* input = reinterpret_cast<const char*>(data);
//https://stackoverflow.com/questions/47604431/why-we-can-mmap-to-a-file-but-exceed-the-file-size

    const auto est_line_width = sample_line_width(input, size);
    cout << "estimated line width: " << est_line_width << std::endl;
    const auto est_record_count = size/est_line_width;
    cout << "estimated line count: " << est_record_count << std::endl;

    const auto num_threads = 2;// TODO std::thread::hardware_concurrency();


    std::vector<int32_t> dest1;
    dest1.resize(est_record_count);




    auto dest_tuple = create_vectors<TupleType>(est_record_count);
    std::vector<worker<TupleType>> workers;


    std::thread threads[num_threads];

    size_t remaining = size;
    size_t partition_size = size / num_threads;
    const char* data_start = static_cast<const char*>(data);
    const char* partition_start = data_start;

    // create workers
    for (unsigned i = 0; i < num_threads; ++i) {
        size_t size_hint = std::min(remaining, partition_size);
        remaining -= size_hint;
        // worker(const char* data_start, const char* partition_start, size_t partition_size_hint, unsigned thread_num)
        workers.emplace_back(data_start, partition_start, size_hint, num_threads, i);
        partition_start += size_hint;
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

    // cleanup
    munmap(data, mapping_size);
    close(handle);
}



/*
struct lineitem_table_t {
    std::vector<uint32_t> l_orderkey;
    std::vector<uint32_t> l_partkey;
    std::vector<uint32_t> l_suppkey;
    std::vector<uint32_t> l_linenumber;
    std::vector<int64_t> l_quantity;
    std::vector<int64_t> l_extendedprice;
    std::vector<int64_t> l_discount;
    std::vector<int64_t> l_tax;
    std::vector<char> l_returnflag;
    std::vector<char> l_linestatus;
    std::vector<uint32_t> l_shipdate;
    std::vector<uint32_t> l_commitdate;
    std::vector<uint32_t> l_receiptdate;
    std::vector<std::array<char, 25>> l_shipinstruct;
    std::vector<std::array<char, 10>> l_shipmode;
    std::vector<std::array<char, 44>> l_comment;
};
*/

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <lineitem.tbl>" << std::endl;
        return 1;
    }

    using lineitem_tuple = std::tuple<
        uint32_t, // l_orderkey
        uint32_t,
        uint32_t,
        uint32_t,
        numeric<15, 2>, // l_quantity
        numeric<15, 2>,
        numeric<15, 2>,
        numeric<15, 2>,
        char, // l_returnflag
        char,
        date, // l_shipdate
        uint32_t,
        uint32_t,
        std::array<char, 25>, // l_shipinstruct
        std::array<char, 10>, // l_shipmode
        std::array<char, 44>  // l_comment
        >;
    parse<lineitem_tuple>(argv[1]);

    return 0;
}
