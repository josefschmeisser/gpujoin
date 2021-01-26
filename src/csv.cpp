//#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include <string>
#include <string_view>
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

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
    // we may assume that reads from 'begin' within [len, len + 8) yield zero
    for (size_t i = 0; i < len; i += 8) {
        uint64_t block = *reinterpret_cast<const uint64_t*>(begin + i);
        uint64_t matches = get_matches(pattern, block);
        if (matches != 0) {
            uint64_t pos = __builtin_ctzll(matches) / 8;
            if (pos < 8) {
                return (i + pos);
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

static std::vector<int64_t> results;


struct worker_state {
};

template <bool first_partition = false>
void sum_extendedprice(const char* data_start, const char* partition_start, size_t partition_size_hint, unsigned thread_num, std::vector<uint64_t>& dst) {
    int64_t sum = 0;

    uint64_t dest_index = thread_num;

    // correct partition size
    if constexpr (!first_partition) {
        size_t offset = 0;
        const size_t max_offset = partition_start - data_start;
        while (offset < partition_size_hint) {
            if (offset > max_offset) {
                return;
            } else if (*(partition_start - offset) == '\n') {
                break;
            } else {
                offset++;
            }
        }
        partition_start -= offset;
        partition_size_hint += offset;
    }

    constexpr uint64_t bar_pattern = 0x7C7C7C7C7C7C7C7Cull;
    constexpr uint64_t newline_pattern = 0x0A0A0A0A0A0A0A0Aull;

    size_t i = 0;
 //   int64_t bar_cnt = 0;
    // for each line
#if 0
    while (i < partition_size_hint) {
        auto pos = find_first(bar_pattern, partition_start + i, partition_size_hint - i);
        bar_cnt += (pos >= 0);
        if (bar_cnt == 5) {
            auto bar_pos = i + pos + 1;
            auto len = find_first(bar_pattern, partition_start + bar_pos, partition_size_hint - bar_pos);
            assert(len >= 1);
            int64_t extendedprice = read_numeric(partition_start + bar_pos, len);
            sum += extendedprice;
            bar_cnt = 0;

            // jump to to the end of the line
            i = bar_pos + len + 1;
            auto newline_pos = find_first(newline_pattern, partition_start + i, partition_size_hint - i);
            if (newline_pos < 0 || (newline_pos + i) > partition_size_hint) {
                // undo
                sum -= extendedprice;
                break;
            }
            i += newline_pos + 1;
        } else {
            i += pos + 1;
        }
    }
#endif

const unsigned column_count = 16;
const auto dest_limit = dst.size();
    while (i < partition_size_hint &&  dest_index < dest_limit) {
    
        // parse column 0
        auto bar_pos = find_first(bar_pattern, partition_start + i, partition_size_hint - i);
        unsigned bar_cnt = (bar_pos >= 0);
        bar_pos = std::max(bar_pos, 0l);
        auto col0 = read_int(partition_start + i, bar_pos);
        dst[dest_index] = col0;
        i += bar_pos + 1;
        // parse column 1
        bar_pos = find_first(bar_pattern, partition_start + i, partition_size_hint - i);
        bar_cnt += (bar_pos >= 0);
        bar_pos = std::max(bar_pos, 0l);
    //    auto col1_begin = i + bar_pos + 1;
        auto col1 = read_int(partition_start + i, bar_pos);


        // TODO

        auto newline_pos = find_first(newline_pattern, partition_start + i, partition_size_hint - i);
        i += newline_pos + 1;

cout << "col0: " << col0 << " col1: " << col1 << std::endl;

/*
        int column0;
        int column1;

        ssize_t bar_pos;
        for (unsigned column = 0; column < column_count; ++column) {
            auto bar_pos = find_first(bar_pattern, partition_start + i, partition_size_hint - i);
            bar_cnt += (bar_pos >= 0);

        }
*/
        if (bar_cnt != 2) {// column_count - 1) {
            std::cerr << "invalid line at char ..." << std::endl;
            return;
        }

    //    break;
    if (i > 300) break;
    }

   // *result = sum;
}



int32_t parse_int(const char* begin);

unsigned sample_line_width(const char* data_start, size_t data_size) {
    constexpr uint64_t newline_pattern = 0x0A0A0A0A0A0A0A0Aull;

    ssize_t newline_pos;
    size_t acc = 0, count = 0;//, last_newline = 0;
    for (size_t i = 0; i < data_size && count < 10; i += newline_pos + 1) {
//cout << "search start: " << i << std::endl;
        newline_pos = find_first(newline_pattern, data_start + i, data_size - i);

//if (i > 0) std::cout << "char: " << (int) data_start[i] << std::endl;

        if (i == 0) continue; // skip first line
        else if (newline_pos < 0) break;
//cout << "line width: " << newline_pos << std::endl;
        acc += newline_pos;
        ++count;
    //    last_newline = newline_pos;
    }

    std::cout << "count: " << count << " acc: " << acc << std::endl;
 //   while (i < partition_size_hint &&  dest_index < dest_limit) {

    return (count > 0) ? acc/count : data_size;
}

void rebalance() {
}

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


/*
    const uint8_t* input = nullptr;
    size_t size = 0;
*/


    const auto est_line_width = sample_line_width(input, size);
cout << "estimated line width: " << est_line_width << std::endl;
    const auto est_record_count = size/est_line_width;
cout << "estimated line count: " << est_record_count << std::endl;

    const auto num_threads = 1;// TODO std::thread::hardware_concurrency();


std::vector<uint64_t> dest1;
dest1.resize(est_record_count);


  //  const auto partition_size = size/num_threads;

    results.resize(num_threads, 0);
    std::thread threads[num_threads];

    size_t remaining = size;
    size_t partition_size = size / num_threads;
    const char* data_start = static_cast<const char*>(data);
    const char* partion_start = data_start;



    for (unsigned i = 0; i < num_threads; ++i) {
        size_t size_hint = std::min(remaining, partition_size);
        remaining -= size_hint;
        int64_t* result = &results[i];
        //void sum_extendedprice(const char* data_start, const char* partition_start, size_t partition_size_hint, unsigned thread_num, std::vector<uint64_t>& dst) 
        if (i == 0) {
            threads[0] = std::thread(&sum_extendedprice<true>, data_start, partion_start, size_hint, i, std::ref(dest1));
        } else {
            threads[i] = std::thread(&sum_extendedprice<false>, data_start, partion_start, size_hint, i, std::ref(dest1));
        }
        partion_start += size_hint;
    }
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }




    // cleanup
    munmap(data, mapping_size);
    close(handle);


    // aggregate results of all workers
    int64_t price_sum = 0;//std::accumulate(results.begin(), results.end(), 0ul);


}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <lineitem.tbl>" << std::endl;
        return 1;
    }

    parse(argv[1]);

    return 0;
}
