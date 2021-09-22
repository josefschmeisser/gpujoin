#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <cassert>
#include <algorithm>

#include "common.hpp"
#include "thirdparty/parser.hpp"

#include "parser.hpp"

using namespace std;
using namespace aria::csv;

//std::unordered_map<uint32_t, size_t> part_partkey_index;

static int64_t to_int64(std::string_view s) {
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

template<unsigned Precision, unsigned Scale>
static numeric<Precision, Scale> to_numeric(std::string_view s) {
/*
    size_t dot_position = s.size() - Scale - 1;
    assert(s[dot_position] == '.');
    auto part1 = s.substr(0, dot_position);
    auto part2 = s.substr(dot_position + 1);
    int64_t value = to_int64(part1) * exp10[Scale & 15] + to_int64(part2);
    return numeric<Precision, Scale>{ value };
*/
    constexpr uint64_t period_pattern = 0x2E2E2E2E2E2E2E2Eull;
    int64_t numeric_raw = 0;
    ssize_t dot_position = find_first(period_pattern, s.data(), s.size());

    if (dot_position < 0) {
        // no dot
        int64_t numeric_raw = exp10[Scale & 15] * to_int(s.substr(0, s.size()));
    } else if (dot_position == 0) {
        auto part2 = s.substr(dot_position + 1); // TODO limit number of digits
        int64_t numeric_raw = to_int(part2);
    } else {
        auto part1 = s.substr(0, dot_position);
        auto part2 = s.substr(dot_position + 1);
        int64_t numeric_raw = to_int(part1) * exp10[Scale & 15] + to_int(part2); // TODO scale and limit number of digits
    }
    return numeric<Precision, Scale>{ numeric_raw };
}

static date to_date(const std::string& date_str) {
    uint32_t day, month, year;
    sscanf(date_str.c_str(), "%4d-%2d-%2d", &year, &month, &day);
    const auto jd = to_julian_day(day, month, year);
    return date{ jd };
}

static void load_lineitem_table_with_aria_parser(const std::string& file_name, lineitem_table_t& table) {
    std::ifstream f(file_name);
    CsvParser lineitem = CsvParser(f).delimiter('|');

    std::array<char, 25> l_shipinstruct;
    std::array<char, 10> l_shipmode;
    std::array<char, 44> l_comment;
    for (auto row : lineitem) {
        table.l_orderkey.push_back(static_cast<uint32_t>(std::stoul(row[0])));
        table.l_partkey.push_back(static_cast<uint32_t>(std::stoul(row[1])));
        table.l_suppkey.push_back(static_cast<uint32_t>(std::stoul(row[2])));
        table.l_linenumber.push_back(static_cast<uint32_t>(std::stoul(row[3])));
        table.l_quantity.push_back(to_numeric<15, 2>(std::string_view(row[4])));
        table.l_extendedprice.push_back(to_numeric<15, 2>(std::string_view(row[5])));
        table.l_discount.push_back(to_numeric<15, 2>(std::string_view(row[6])));
        table.l_tax.push_back(to_numeric<15, 2>(std::string_view(row[7])));
        table.l_returnflag.push_back(row[8][0]);
        table.l_linestatus.push_back(row[9][0]);
        table.l_shipdate.push_back(to_date(row[10]));
        table.l_commitdate.push_back(to_date(row[11]));
        table.l_receiptdate.push_back(to_date(row[12]));
        std::strncpy(l_shipinstruct.data(), row[13].c_str(),
                     sizeof(l_shipinstruct));
        table.l_shipinstruct.push_back(l_shipinstruct);
        std::strncpy(l_shipmode.data(), row[14].c_str(), sizeof(l_shipmode));
        table.l_shipmode.push_back(l_shipmode);
        std::strncpy(l_comment.data(), row[15].c_str(), sizeof(l_comment));
        table.l_comment.push_back(l_comment);
    }
}

static void load_lineitem_table(const std::string& file_name, lineitem_table_t& table) {
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
        date,
        date,
        std::array<char, 25>, // l_shipinstruct
        std::array<char, 10>, // l_shipmode
        std::array<char, 44>  // l_comment
        >;
    auto result = parse<lineitem_tuple, table_allocator>(file_name);
    table.l_orderkey.swap(*std::get<0>(result));
    table.l_partkey.swap(*std::get<1>(result));
    table.l_suppkey.swap(*std::get<2>(result));
    table.l_linenumber.swap(*std::get<3>(result));
    table.l_quantity.swap(*std::get<4>(result));
    table.l_extendedprice.swap(*std::get<5>(result));
    table.l_discount.swap(*std::get<6>(result));
    table.l_tax.swap(*std::get<7>(result));
    table.l_returnflag.swap(*std::get<8>(result));
    table.l_linestatus.swap(*std::get<9>(result));
    table.l_shipdate.swap(*std::get<10>(result));
    table.l_commitdate.swap(*std::get<11>(result));
    table.l_receiptdate.swap(*std::get<12>(result));
    table.l_shipinstruct.swap(*std::get<13>(result));
    table.l_shipmode.swap(*std::get<14>(result));
    table.l_comment.swap(*std::get<15>(result));
}

static void load_part_table_with_aria_parser(const std::string& file_name, part_table_t& table) {
    std::ifstream f(file_name);
    CsvParser lineitem = CsvParser(f).delimiter('|');

    std::array<char, 55> p_name;
    std::array<char, 25> p_mfgr;
    std::array<char, 10> p_brand;
    std::array<char, 25> p_type;
    std::array<char, 10> p_container;
    std::array<char, 23> p_comment;

    size_t tid = 0;
    for (auto row : lineitem) {
        table.p_partkey.push_back(static_cast<uint32_t>(std::stoul(row[0])));
        std::strncpy(p_name.data(), row[1].c_str(), sizeof(p_name));
        table.p_name.push_back(p_name);
        std::strncpy(p_mfgr.data(), row[2].c_str(), sizeof(p_mfgr));
        table.p_mfgr.push_back(p_mfgr);
        std::strncpy(p_brand.data(), row[3].c_str(), sizeof(p_brand));
        table.p_brand.push_back(p_brand);
        std::strncpy(p_type.data(), row[4].c_str(), sizeof(p_type));
        table.p_type.push_back(p_type);
        table.p_size.push_back(std::stoi(row[5]));
        std::strncpy(p_container.data(), row[6].c_str(), sizeof(p_container));
        table.p_container.push_back(p_container);
        table.p_retailprice.push_back(to_numeric<15, 2>(std::string_view(row[7])));
        std::strncpy(p_comment.data(), row[8].c_str(), sizeof(p_comment));
        table.p_comment.push_back(p_comment);

        // add index entry
        //part_partkey_index[table.p_partkey.back()] = tid++;
    }
}

static void load_part_table(const std::string& file_name, part_table_t& table) {
    using part_tuple = std::tuple<
        uint32_t, // p_partkey
        std::array<char, 55>,
        std::array<char, 25>,
        std::array<char, 10>,
        std::array<char, 25>,
        int32_t,
        std::array<char, 10>,
        numeric<15, 2>,
        std::array<char, 23>
        >;
    auto result = parse<part_tuple, table_allocator>(file_name);
    table.p_partkey.swap(*std::get<0>(result));
    table.p_name.swap(*std::get<1>(result));
    table.p_mfgr.swap(*std::get<2>(result));
    table.p_brand.swap(*std::get<3>(result));
    table.p_type.swap(*std::get<4>(result));
    table.p_size.swap(*std::get<5>(result));
    table.p_container.swap(*std::get<6>(result));
    table.p_retailprice.swap(*std::get<7>(result));
    table.p_comment.swap(*std::get<8>(result));
}

void load_tables(Database& db, const std::string& path) {
    load_lineitem_table(path + "lineitem.tbl", db.lineitem);
    load_part_table(path + "part.tbl", db.part);
}

void load_tables_with_aria_parser(Database& db, const std::string& path) {
    load_lineitem_table_with_aria_parser(path + "lineitem.tbl", db.lineitem);
    load_part_table_with_aria_parser(path + "part.tbl", db.part);
}

#if false
void prepareManaged(lineitem_table_t& src, lineitem_table_plain_t& dst) {
    const auto N = src.l_commitdate.size();

    size_t columnSize = N*sizeof(decltype(src.l_orderkey)::value_type);
    cudaMallocManaged(&dst.l_orderkey, columnSize);
    std::memcpy(dst.l_orderkey, src.l_orderkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_partkey)::value_type);
    cudaMallocManaged(&dst.l_partkey, columnSize);
    std::memcpy(dst.l_partkey, src.l_partkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_suppkey)::value_type);
    cudaMallocManaged(&dst.l_suppkey, columnSize);
    std::memcpy(dst.l_suppkey, src.l_suppkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_linenumber)::value_type);
    cudaMallocManaged(&dst.l_linenumber, columnSize);
    std::memcpy(dst.l_linenumber, src.l_linenumber.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_quantity)::value_type);
    cudaMallocManaged(&dst.l_quantity, columnSize);
    std::memcpy(dst.l_quantity, src.l_quantity.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_extendedprice)::value_type);
    cudaMallocManaged(&dst.l_extendedprice, columnSize);
    std::memcpy(dst.l_extendedprice, src.l_extendedprice.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_discount)::value_type);
    cudaMallocManaged(&dst.l_discount, columnSize);
    std::memcpy(dst.l_discount, src.l_discount.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_tax)::value_type);
    cudaMallocManaged(&dst.l_tax, columnSize);
    std::memcpy(dst.l_tax, src.l_tax.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_returnflag)::value_type);
    cudaMallocManaged(&dst.l_returnflag, columnSize);
    std::memcpy(dst.l_returnflag, src.l_returnflag.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_linestatus)::value_type);
    cudaMallocManaged(&dst.l_linestatus, columnSize);
    std::memcpy(dst.l_linestatus, src.l_linestatus.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_shipdate)::value_type);
    cudaMallocManaged(&dst.l_shipdate, columnSize);
    std::memcpy(dst.l_shipdate, src.l_shipdate.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_commitdate)::value_type);
    cudaMallocManaged(&dst.l_commitdate, columnSize);
    std::memcpy(dst.l_commitdate, src.l_commitdate.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_receiptdate)::value_type);
    cudaMallocManaged(&dst.l_receiptdate, columnSize);
    std::memcpy(dst.l_receiptdate, src.l_receiptdate.data(), columnSize);
/*
    columnSize = N*sizeof(decltype(src.l_shipinstruct)::value_type);
    cudaMallocManaged(&dst.l_shipinstruct, columnSize);
    std::memcpy(dst.l_shipinstruct, src.l_shipinstruct.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_shipmode)::value_type);
    cudaMallocManaged(&dst.l_shipmode, columnSize);
    std::memcpy(dst.l_shipmode, src.l_shipmode.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_comment)::value_type);
    cudaMallocManaged(&dst.l_comment, columnSize);
    std::memcpy(dst.l_comment, src.l_comment.data(), columnSize);
*/
}

void prepareDeviceResident(lineitem_table_t& src, lineitem_table_plain_t& dst) {
    const auto N = src.l_commitdate.size();

    size_t columnSize = N*sizeof(decltype(src.l_orderkey)::value_type);
    cudaMalloc((void**)&dst.l_orderkey, columnSize);
    cudaMemcpy(dst.l_orderkey, src.l_orderkey.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_partkey)::value_type);
    cudaMalloc((void**)&dst.l_partkey, columnSize);
    cudaMemcpy(dst.l_partkey, src.l_partkey.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_suppkey)::value_type);
    cudaMalloc((void**)&dst.l_suppkey, columnSize);
    cudaMemcpy(dst.l_suppkey, src.l_suppkey.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_linenumber)::value_type);
    cudaMalloc((void**)&dst.l_linenumber, columnSize);
    cudaMemcpy(dst.l_linenumber, src.l_linenumber.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_quantity)::value_type);
    cudaMalloc((void**)&dst.l_quantity, columnSize);
    cudaMemcpy(dst.l_quantity, src.l_quantity.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_extendedprice)::value_type);
    cudaMalloc((void**)&dst.l_extendedprice, columnSize);
    cudaMemcpy(dst.l_extendedprice, src.l_extendedprice.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_discount)::value_type);
    cudaMalloc((void**)&dst.l_discount, columnSize);
    cudaMemcpy(dst.l_discount, src.l_discount.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_tax)::value_type);
    cudaMalloc((void**)&dst.l_tax, columnSize);
    cudaMemcpy(dst.l_tax, src.l_tax.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_returnflag)::value_type);
    cudaMalloc((void**)&dst.l_returnflag, columnSize);
    cudaMemcpy(dst.l_returnflag, src.l_returnflag.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_linestatus)::value_type);
    cudaMalloc((void**)&dst.l_linestatus, columnSize);
    cudaMemcpy(dst.l_linestatus, src.l_linestatus.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_shipdate)::value_type);
    cudaMalloc((void**)&dst.l_shipdate, columnSize);
    cudaMemcpy(dst.l_shipdate, src.l_shipdate.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_commitdate)::value_type);
    cudaMalloc((void**)&dst.l_commitdate, columnSize);
    cudaMemcpy(dst.l_commitdate, src.l_commitdate.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_receiptdate)::value_type);
    cudaMalloc((void**)&dst.l_receiptdate, columnSize);
    cudaMemcpy(dst.l_receiptdate, src.l_receiptdate.data(), columnSize, cudaMemcpyHostToDevice);

/*
    columnSize = N*sizeof(decltype(src.l_shipinstruct)::value_type);

    columnSize = N*sizeof(decltype(src.l_shipmode)::value_type);

    columnSize = N*sizeof(decltype(src.l_comment)::value_type);
*/
}
#endif

void sort_relation(part_table_t& part) {
    auto permutation = compute_permutation(part.p_partkey.begin(), part.p_partkey.end(), std::less<>{});
    apply_permutation(permutation, part.p_partkey, part.p_name, part.p_mfgr, part.p_brand, part.p_type, part.p_size, part.p_container, part.p_retailprice, part.p_comment);
}

void sort_relation(lineitem_table_t& lineitem) {
    auto permutation = compute_permutation(lineitem.l_orderkey.begin(), lineitem.l_orderkey.end(), std::less<>{});
    apply_permutation(
        permutation,
        lineitem.l_orderkey,
        lineitem.l_partkey,
        lineitem.l_suppkey,
        lineitem.l_linenumber,
        lineitem.l_quantity,
        lineitem.l_extendedprice,
        lineitem.l_discount,
        lineitem.l_tax,
        lineitem.l_returnflag,
        lineitem.l_linestatus,
        lineitem.l_shipdate,
        lineitem.l_commitdate,
        lineitem.l_receiptdate,
        lineitem.l_shipinstruct,
        lineitem.l_shipmode,
        lineitem.l_comment
    );
}
