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

template<class T> using table_allocator = std::allocator<T>;

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

template<class T>
struct to_raw_type {
    using type = T;
};

template<unsigned Precision, unsigned Scale>
struct to_raw_type<numeric<Precision, Scale>> {
    using type = typename numeric<Precision, Scale>::raw_type;
};

template<>
struct to_raw_type<date> {
    using type = date::raw_type;
};

template<class T>
struct column {
    using value_type = T;
    using raw_type = typename to_raw_type<T>::type;
    using plain_array_type = typename to_raw_type<T>::type *;
    using vector_type = std::vector<T, table_allocator<T>>;
};

template<class T>
using column_raw_t = typename column<T>::raw_type;

template<class VectorType>
using vector_to_raw_t = typename column<typename VectorType::value_type>::raw_type;

struct lineitem_table_t {
    column<uint32_t>::vector_type l_orderkey;
    column<uint32_t>::vector_type l_partkey;
    //column<uint64_t>::vector_type l_partkey;
    column<uint32_t>::vector_type l_suppkey;
    column<uint32_t>::vector_type l_linenumber;
    column<numeric<15, 2>>::vector_type l_quantity;
    column<numeric<15, 2>>::vector_type l_extendedprice;
    column<numeric<15, 2>>::vector_type l_discount;
    column<numeric<15, 2>>::vector_type l_tax;
    column<char>::vector_type l_returnflag;
    column<char>::vector_type l_linestatus;
    column<date>::vector_type l_shipdate;
    column<date>::vector_type l_commitdate;
    column<date>::vector_type l_receiptdate;
    column<std::array<char, 25>>::vector_type l_shipinstruct;
    column<std::array<char, 10>>::vector_type l_shipmode;
    column<std::array<char, 44>>::vector_type l_comment;
};
/*
struct lineitem_table_plain_t {
    uint32_t* l_orderkey;
    //uint32_t* l_partkey;
    uint64_t* l_partkey;
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
};*/
struct lineitem_table_plain_t {
    column<decltype(lineitem_table_t::l_orderkey)::value_type>::plain_array_type l_orderkey;
    column<decltype(lineitem_table_t::l_partkey)::value_type>::plain_array_type l_partkey;
    column<decltype(lineitem_table_t::l_suppkey)::value_type>::plain_array_type l_suppkey;
    column<decltype(lineitem_table_t::l_linenumber)::value_type>::plain_array_type l_linenumber;
    column<decltype(lineitem_table_t::l_quantity)::value_type>::plain_array_type l_quantity;
    column<decltype(lineitem_table_t::l_extendedprice)::value_type>::plain_array_type l_extendedprice;
    column<decltype(lineitem_table_t::l_discount)::value_type>::plain_array_type l_discount;
    column<decltype(lineitem_table_t::l_tax)::value_type>::plain_array_type l_tax;
    column<decltype(lineitem_table_t::l_returnflag)::value_type>::plain_array_type l_returnflag;
    column<decltype(lineitem_table_t::l_linestatus)::value_type>::plain_array_type l_linestatus;
    column<decltype(lineitem_table_t::l_shipdate)::value_type>::plain_array_type l_shipdate;
    column<decltype(lineitem_table_t::l_commitdate)::value_type>::plain_array_type l_commitdate;
    column<decltype(lineitem_table_t::l_receiptdate)::value_type>::plain_array_type l_receiptdate;
    column<decltype(lineitem_table_t::l_shipinstruct)::value_type>::plain_array_type l_shipinstruct;
    column<decltype(lineitem_table_t::l_shipmode)::value_type>::plain_array_type l_shipmode;
    column<decltype(lineitem_table_t::l_comment)::value_type>::plain_array_type l_comment;
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
    column<uint32_t>::vector_type p_partkey;
    //column<uint64_t>::vector_type p_partkey;
    column<std::array<char, 55>>::vector_type p_name;
    column<std::array<char, 25>>::vector_type p_mfgr;
    column<std::array<char, 10>>::vector_type p_brand;
    column<std::array<char, 25>>::vector_type p_type;
    column<int32_t>::vector_type p_size;
    column<std::array<char, 10>>::vector_type p_container;
    column<numeric<15, 2>>::vector_type p_retailprice;
    column<std::array<char, 23>>::vector_type p_comment;
};
/*
struct part_table_plain_t {
    //uint32_t* p_partkey;
    uint64_t* p_partkey;
    std::array<char, 55>* p_name;
    std::array<char, 25>* p_mfgr;
    std::array<char, 10>* p_brand;
    std::array<char, 25>* p_type;
    int32_t* p_size;
    std::array<char, 10>* p_container;
    int64_t* p_retailprice;
    std::array<char, 23>* p_comment;
};*/
struct part_table_plain_t {
    column<decltype(part_table_t::p_partkey)::value_type>::plain_array_type p_partkey;
    column<decltype(part_table_t::p_name)::value_type>::plain_array_type p_name;
    column<decltype(part_table_t::p_mfgr)::value_type>::plain_array_type p_mfgr;
    column<decltype(part_table_t::p_brand)::value_type>::plain_array_type p_brand;
    column<decltype(part_table_t::p_type)::value_type>::plain_array_type p_type;
    column<decltype(part_table_t::p_size)::value_type>::plain_array_type p_size;
    column<decltype(part_table_t::p_container)::value_type>::plain_array_type p_container;
    column<decltype(part_table_t::p_retailprice)::value_type>::plain_array_type p_retailprice;
    column<decltype(part_table_t::p_comment)::value_type>::plain_array_type p_comment;
};

struct Database {
    lineitem_table_t lineitem;
    part_table_t part;
};

void load_tables(Database& db, const std::string& path);

void load_tables_with_aria_parser(Database& db, const std::string& path);

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
    return std::make_pair(dst, std::move(plain));
}

template<class TargetAllocator>
auto migrate_relation(lineitem_table_t& src, TargetAllocator& target_allocator) {
    const auto N = src.l_orderkey.size();
    auto plain = std::make_unique<lineitem_table_plain_t>();

    // copy columns
    plain->l_orderkey = create_device_array_from(src.l_orderkey, target_allocator).release();
    plain->l_partkey = create_device_array_from(src.l_partkey, target_allocator).release();
    plain->l_suppkey = create_device_array_from(src.l_suppkey, target_allocator).release();
    plain->l_linenumber = create_device_array_from(src.l_linenumber, target_allocator).release();
    plain->l_quantity = reinterpret_cast<typename decltype(src.l_quantity)::value_type::raw_type*>(create_device_array_from(src.l_quantity, target_allocator).release());
    plain->l_extendedprice = reinterpret_cast<typename decltype(src.l_extendedprice)::value_type::raw_type*>(create_device_array_from(src.l_extendedprice, target_allocator).release());
    plain->l_discount = reinterpret_cast<typename decltype(src.l_discount)::value_type::raw_type*>(create_device_array_from(src.l_discount, target_allocator).release());
    plain->l_tax = reinterpret_cast<typename decltype(src.l_tax)::value_type::raw_type*>(create_device_array_from(src.l_tax, target_allocator).release());
    plain->l_returnflag = create_device_array_from(src.l_returnflag, target_allocator).release();
    plain->l_linestatus = create_device_array_from(src.l_linestatus, target_allocator).release();
    plain->l_shipdate = reinterpret_cast<typename decltype(src.l_shipdate)::value_type::raw_type*>(create_device_array_from(src.l_shipdate, target_allocator).release());
    plain->l_commitdate = reinterpret_cast<typename decltype(src.l_commitdate)::value_type::raw_type*>(create_device_array_from(src.l_commitdate, target_allocator).release());
    plain->l_receiptdate = reinterpret_cast<typename decltype(src.l_receiptdate)::value_type::raw_type*>(create_device_array_from(src.l_receiptdate, target_allocator).release());
    plain->l_shipinstruct = create_device_array_from(src.l_shipinstruct, target_allocator).release();
    plain->l_shipmode = create_device_array_from(src.l_shipmode, target_allocator).release();
    plain->l_comment = create_device_array_from(src.l_comment, target_allocator).release();

    lineitem_table_plain_t* dst;
    cudaMalloc(&dst, sizeof(lineitem_table_plain_t));
    cudaMemcpy(dst, plain.get(), sizeof(lineitem_table_plain_t), cudaMemcpyHostToDevice);
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
    return std::make_pair(dst, std::move(plain));
}

template<class TargetAllocator>
auto migrate_relation(part_table_t& src, TargetAllocator& target_allocator) {
    const auto N = src.p_partkey.size();
    auto plain = std::make_unique<part_table_plain_t>();

    // copy columns
    plain->p_partkey = create_device_array_from(src.p_partkey, target_allocator).release();
    plain->p_name = create_device_array_from(src.p_name, target_allocator).release();
    plain->p_mfgr = create_device_array_from(src.p_mfgr, target_allocator).release();
    plain->p_brand = create_device_array_from(src.p_brand, target_allocator).release();
    plain->p_type = create_device_array_from(src.p_type, target_allocator).release();
    plain->p_size = create_device_array_from(src.p_size, target_allocator).release();
    plain->p_container = create_device_array_from(src.p_container, target_allocator).release();
    plain->p_retailprice = reinterpret_cast<typename decltype(src.p_retailprice)::value_type::raw_type*>(create_device_array_from(src.p_retailprice, target_allocator).release());
    plain->p_comment = create_device_array_from(src.p_comment, target_allocator).release();

    part_table_plain_t* dst;
    cudaMalloc(&dst, sizeof(part_table_plain_t));
    cudaMemcpy(dst, plain.get(), sizeof(part_table_plain_t), cudaMemcpyHostToDevice);
    return std::make_pair(dst, std::move(plain));
}

void sort_relation(part_table_t& part);

void sort_relation(lineitem_table_t& lineitem);
