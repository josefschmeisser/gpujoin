#include <bits/stdint-intn.h>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <cassert>

#include "common.hpp"
#include "thirdparty/parser.hpp"

using namespace aria::csv;

std::unordered_map<uint32_t, size_t> part_partkey_index;

static int64_t to_int(std::string_view s) {
    int64_t result = 0;
    for (auto c : s) result = result * 10 + (c - '0');
    return result;
}

static constexpr int64_t exp10[] = {
    1ul,
    10ul,
    100ul,
    1000ul,
    10000ul,
    100000ul,
    1000000ul,
    10000000ul,
    100000000ul,
    1000000000ul,
    10000000000ul,
    100000000000ul,
    1000000000000ul,
    10000000000000ul,
    100000000000000ul,
};

static int64_t to_numeric(std::string_view s, size_t scale) {
    size_t dot_position = s.size() - scale - 1;
    assert(s[dot_position] == '.');
    auto part1 = s.substr(0, dot_position);
    auto part2 = s.substr(dot_position + 1);
    int64_t value = to_int(part1) * exp10[scale & 15] + to_int(part2);
    return value;
}

// source:
// https://stason.org/TULARC/society/calendars/2-15-1-Is-there-a-formula-for-calculating-the-Julian-day-nu.html
constexpr uint32_t to_julian_day(uint32_t day, uint32_t month, uint32_t year) {
    uint32_t a = (14 - month) / 12;
    uint32_t y = year + 4800 - a;
    uint32_t m = month + 12 * a - 3;
    return day + (153 * m + 2) / 5 + y * 365 + y / 4 - y / 100 + y / 400 -
           32045;
}

static uint32_t to_julian_day(const std::string& date) {
    uint32_t day, month, year;
    sscanf(date.c_str(), "%4d-%2d-%2d", &year, &month, &day);
    return to_julian_day(day, month, year);
}

static void load_lineitem_table(const std::string& file_name, lineitem_table_t& table) {
    std::ifstream f(file_name);
    CsvParser lineitem = CsvParser(f).delimiter('|');

    std::array<char, 25> l_shipinstruct;
    std::array<char, 10> l_shipmode;
    for (auto row : lineitem) {
        table.l_orderkey.push_back(static_cast<uint32_t>(std::stoul(row[0])));
        table.l_partkey.push_back(static_cast<uint32_t>(std::stoul(row[1])));
        table.l_suppkey.push_back(static_cast<uint32_t>(std::stoul(row[2])));
        table.l_linenumber.push_back(static_cast<uint32_t>(std::stoul(row[3])));
        table.l_quantity.push_back(to_int(std::string_view(row[4])));
        table.l_extendedprice.push_back(to_numeric(std::string_view(row[5]), 2));
        table.l_discount.push_back(to_numeric(std::string_view(row[6]), 2));
        table.l_tax.push_back(to_numeric(std::string_view(row[7]), 2));
        table.l_returnflag.push_back(row[8][0]);
        table.l_linestatus.push_back(row[9][0]);
        table.l_shipdate.push_back(to_julian_day(row[10]));
        table.l_commitdate.push_back(to_julian_day(row[11]));
        table.l_receiptdate.push_back(to_julian_day(row[12]));
        std::strncpy(l_shipinstruct.data(), row[13].c_str(),
                     sizeof(l_shipinstruct));
        table.l_shipinstruct.push_back(l_shipinstruct);
        std::strncpy(l_shipmode.data(), row[14].c_str(), sizeof(l_shipmode));
        table.l_shipmode.push_back(l_shipmode);
        table.l_comment.push_back(row[15]);
    }
}

static void load_part_table(const std::string& file_name, part_table_t& table) {
    std::ifstream f(file_name);
    CsvParser lineitem = CsvParser(f).delimiter('|');

    std::array<char, 25> p_mfgr;
    std::array<char, 10> p_brand;
    std::array<char, 10> p_container;

    size_t tid = 0;
    for (auto row : lineitem) {
        table.p_partkey.push_back(static_cast<uint32_t>(std::stoul(row[0])));
        table.p_name.push_back(row[1]);
        std::strncpy(p_mfgr.data(), row[2].c_str(), sizeof(p_mfgr));
        table.p_mfgr.push_back(p_mfgr);
        std::strncpy(p_brand.data(), row[3].c_str(), sizeof(p_brand));
        table.p_brand.push_back(p_brand);
        table.p_type.push_back(row[4]);
        table.p_size.push_back(std::stoi(row[5]));
        std::strncpy(p_container.data(), row[6].c_str(), sizeof(p_container));
        table.p_container.push_back(p_container);
        table.p_retailprice.push_back(to_numeric(std::string_view(row[7]), 2));
        table.p_comment.push_back(row[8]);

        // add index entry
        part_partkey_index[table.p_partkey.back()] = tid++;
    }
}

void load_tables(Database& db, const std::string& path) {
    load_lineitem_table(path + "lineitem.tbl", db.lineitem);
    load_part_table(path + "part.tbl", db.part);
}

static inline uint64_t ilog2(uint64_t num) {
    uint64_t lz = __builtin_clzl(num);
    return lz^63;
}

static inline int64_t func(int64_t num, int64_t value) {
    int64_t arg = (num + 1)*value;
    return ilog2(arg);
}

/*
select
        100.00 * sum(case
                when p_type like 'PROMO%'
                        then l_extendedprice * func(l_discount, <value>)
                else 0
        end) / sum(l_extendedprice * func(l_discount, <value>)) as result
from
        lineitem,
        part
where
        l_partkey = p_partkey
        and l_shipdate >= date '1995-09-01'
        and l_shipdate < date '1995-10-01'
*/
void query_14(Database& db) {
    int64_t value = 2;
    int64_t sum1 = 0;
    int64_t sum2 = 0;

    // your code goes here
    constexpr std::string_view prefix = "PROMO";
    auto& part = db.part;
    auto& lineitem = db.lineitem;

    // aggregation loop
    for (size_t i = 0; i < lineitem.l_partkey.size(); ++i) {
        constexpr auto lower_shipdate =
            to_julian_day(1, 9, 1995);  // 1995-09-01
        constexpr auto upper_shipdate =
            to_julian_day(1, 10, 1995);  // 1995-10-01
        if (lineitem.l_shipdate[i] < lower_shipdate ||
            lineitem.l_shipdate[i] >= upper_shipdate) {
            continue;
        }

        // probe
        auto it = part_partkey_index.find(lineitem.l_partkey[i]);
        if (it == part_partkey_index.end()) {
            continue;
        }
        size_t j = it->second;

        auto extendedprice = lineitem.l_extendedprice[i];
        auto discount = lineitem.l_discount[i];
        auto summand = extendedprice * func(discount, value);
        sum2 += summand;

        auto& type = part.p_type[j];
        if (type.compare(0, prefix.size(), prefix) == 0) {
            sum1 += summand;
        }
    }

    int64_t result = 100*sum1/sum2;
    printf("%ld\n", result);
}
