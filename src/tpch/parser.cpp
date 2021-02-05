#include "parser.hpp"

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

ostream& operator<<(ostream& os, const date& value) {
    uint32_t year, month, day;
    input_parser<date>::from_julian_day(value.raw, year, month, day);
    char output[16];
    snprintf(output, sizeof(output), "%04d-%02d-%02d", year, month, day);
    os << output;
    return os;
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
