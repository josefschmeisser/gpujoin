#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

struct lineitem_table_device_t {
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
    std::vector<int64_t> p_retailprice;
    std::vector<std::array<char, 23>> p_comment;
};

struct part_table_device_t {
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
