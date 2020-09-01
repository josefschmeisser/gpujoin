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
    std::vector<std::string> l_comment;
};

struct lineitem_table_mgd_t {
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
    std::array<char, 25> l_shipinstruct;
    std::array<char, 10> l_shipmode;
    std::string l_comment;
};

struct part_table_t {
    std::vector<uint32_t> p_partkey;
    std::vector<std::string> p_name;
    std::vector<std::array<char, 25>> p_mfgr;
    std::vector<std::array<char, 10>> p_brand;
    std::vector<std::string> p_type;
    std::vector<int32_t> p_size;
    std::vector<std::array<char, 10>> p_container;
    std::vector<int64_t> p_retailprice;
    std::vector<std::string> p_comment;
};

struct part_table_mgd_t {
    uint32_t* p_partkey;
    std::string p_name;
    std::array<char, 25> p_mfgr;
    std::array<char, 10> p_brand;
    std::string p_type;
    int32_t* p_size;
    std::array<char, 10> p_container;
    int64_t* p_retailprice;
    std::string p_comment;
};

struct Database {
    lineitem_table_t lineitem;
    part_table_t part;
};

void load_tables(Database& db, const std::string& path);

void query_1(Database& db);

void query_14(Database& db);
