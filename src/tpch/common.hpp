#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <cstring>
#include <type_traits>
#include <vector>
#include <numeric>

#include "utils.hpp"

template<unsigned Precision, unsigned Scale>
struct numeric {
    static constexpr auto precision = Precision;
    static constexpr auto scale = Scale;
    using my_type = numeric<Precision, Scale>;
    using raw_type = int64_t;
    raw_type raw;
};
static_assert(sizeof(numeric<0, 0>) == sizeof(numeric<0, 0>::raw_type));

struct date {
    using raw_type = uint32_t;
    raw_type raw;
};
static_assert(sizeof(date) == sizeof(date::raw_type));

struct lineitem_table_t {
    std::vector<uint32_t> l_orderkey;
    std::vector<uint32_t> l_partkey;
    std::vector<uint32_t> l_suppkey;
    std::vector<uint32_t> l_linenumber;
    std::vector<numeric<15, 2>> l_quantity;
    std::vector<numeric<15, 2>> l_extendedprice;
    std::vector<numeric<15, 2>> l_discount;
    std::vector<numeric<15, 2>> l_tax;
    std::vector<char> l_returnflag;
    std::vector<char> l_linestatus;
    std::vector<date> l_shipdate;
    std::vector<date> l_commitdate;
    std::vector<date> l_receiptdate;
    std::vector<std::array<char, 25>> l_shipinstruct;
    std::vector<std::array<char, 10>> l_shipmode;
    std::vector<std::array<char, 44>> l_comment;
};

struct lineitem_table_plain_t {
    uint32_t* l_orderkey;
    uint32_t* l_partkey;
    uint32_t* l_suppkey;
    uint32_t* l_linenumber;
    int64_t* l_quantity;
    int64_t* l_extendedprice;
    int64_t* l_discount;
    int64_t* l_tax;
    char* l_returnflag;
    char* l_linestatus;
    uint32_t* l_shipdate;
    uint32_t* l_commitdate;
    uint32_t* l_receiptdate;
    std::array<char, 25>* l_shipinstruct;
    std::array<char, 10>* l_shipmode;
    std::array<char, 44>* l_comment;
};

/*
create table part
  (
     p_partkey     integer not null,
     p_name        varchar(55) not null,
     p_mfgr        char(25) not null,
     p_brand       char(10) not null,
     p_type        varchar(25) not null,
     p_size        integer not null,
     p_container   char(10) not null,
     p_retailprice numeric(15, 2) not null,
     p_comment     varchar(23) not null
  );*/

struct part_table_t {
    std::vector<uint32_t> p_partkey;
    std::vector<std::array<char, 55>> p_name;
    std::vector<std::array<char, 25>> p_mfgr;
    std::vector<std::array<char, 10>> p_brand;
    std::vector<std::array<char, 25>> p_type;
    std::vector<int32_t> p_size;
    std::vector<std::array<char, 10>> p_container;
    std::vector<numeric<15, 2>> p_retailprice;
    std::vector<std::array<char, 23>> p_comment;
};

struct part_table_plain_t {
    uint32_t* p_partkey;
    std::array<char, 55>* p_name;
    std::array<char, 25>* p_mfgr;
    std::array<char, 10>* p_brand;
    std::array<char, 25>* p_type;
    int32_t* p_size;
    std::array<char, 10>* p_container;
    int64_t* p_retailprice;
    std::array<char, 23>* p_comment;
};

struct Database {
    lineitem_table_t lineitem;
    part_table_t part;
};

void load_tables(Database& db, const std::string& path);

void query_1(Database& db);

void query_14_part_build(Database& db);

void query_14_lineitem_build(Database& db);

// source:
// https://stason.org/TULARC/society/calendars/2-15-1-Is-there-a-formula-for-calculating-the-Julian-day-nu.html
constexpr uint32_t to_julian_day(uint32_t day, uint32_t month, uint32_t year) {
    uint32_t a = (14 - month) / 12;
    uint32_t y = year + 4800 - a;
    uint32_t m = month + 12 * a - 3;
    return day + (153 * m + 2) / 5 + y * 365 + y / 4 - y / 100 + y / 400 -
           32045;
}

template<class F>
auto copy_relation(const lineitem_table_t& src) {
    const auto N = src.l_orderkey.size();
    static F f;
    auto plain = std::make_unique<lineitem_table_plain_t>();
    plain->l_orderkey = f(src.l_orderkey);
    plain->l_partkey = f(src.l_partkey);
    plain->l_suppkey = f(src.l_suppkey);
    plain->l_linenumber = f(src.l_linenumber);
    plain->l_quantity = reinterpret_cast<typename decltype(src.l_quantity)::value_type::raw_type*>(f(src.l_quantity));
    plain->l_extendedprice = reinterpret_cast<typename decltype(src.l_extendedprice)::value_type::raw_type*>(f(src.l_extendedprice));
    plain->l_discount = reinterpret_cast<typename decltype(src.l_discount)::value_type::raw_type*>(f(src.l_discount));
    plain->l_tax = reinterpret_cast<typename decltype(src.l_tax)::value_type::raw_type*>(f(src.l_tax));
    plain->l_returnflag = f(src.l_returnflag);
    plain->l_linestatus = f(src.l_linestatus);
    plain->l_shipdate = reinterpret_cast<typename decltype(src.l_shipdate)::value_type::raw_type*>(f(src.l_shipdate));
    plain->l_commitdate = reinterpret_cast<typename decltype(src.l_commitdate)::value_type::raw_type*>(f(src.l_commitdate));
    plain->l_receiptdate = reinterpret_cast<typename decltype(src.l_receiptdate)::value_type::raw_type*>(f(src.l_receiptdate));
    plain->l_shipinstruct = f(src.l_shipinstruct);
    plain->l_shipmode = f(src.l_shipmode);
    plain->l_comment = f(src.l_comment);

    lineitem_table_plain_t* dst;

    cudaMalloc(&dst, sizeof(lineitem_table_plain_t));
    cudaMemcpy(dst, plain.get(), sizeof(lineitem_table_plain_t), cudaMemcpyHostToDevice);
/*
    cudaMallocManaged(&dst, sizeof(lineitem_table_plain_t));
    std::memcpy(dst, &tmp, sizeof(lineitem_table_plain_t));*/
    return std::make_pair(dst, std::move(plain));
}

template<class F>
auto copy_relation(const part_table_t& src) {
    const auto N = src.p_partkey.size();
    static F f;
    auto plain = std::make_unique<part_table_plain_t>();
    plain->p_partkey = f(src.p_partkey);
    plain->p_name = f(src.p_name);
    plain->p_mfgr = f(src.p_mfgr);
    plain->p_brand = f(src.p_brand);
    plain->p_type = f(src.p_type);
    plain->p_size = f(src.p_size);
    plain->p_container = f(src.p_container);
    plain->p_retailprice = reinterpret_cast<typename decltype(src.p_retailprice)::value_type::raw_type*>(f(src.p_retailprice));
    plain->p_comment = f(src.p_comment);

    part_table_plain_t* dst;

    cudaMalloc(&dst, sizeof(part_table_plain_t));
    cudaMemcpy(dst, plain.get(), sizeof(part_table_plain_t), cudaMemcpyHostToDevice);
/*
    cudaMallocManaged(&dst, sizeof(part_table_plain_t));
    std::memcpy(dst, &tmp, sizeof(part_table_plain_t));*/
    return std::make_pair(dst, std::move(plain));
}

void sort_relation(part_table_t& part);

void sort_relation(lineitem_table_t& lineitem);
