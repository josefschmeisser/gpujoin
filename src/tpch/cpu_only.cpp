#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cassert>
#include <algorithm>

#include "common.hpp"

using namespace std;

/*
-- TPC-H Query 1

select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus
*/
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
    std::vector<std::string> l_comment;
};

*/
void query_1(Database& db) {
    constexpr auto threshold_date = to_julian_day(2, 9, 1998); // 1998-09-02
    //printf("ship: %d\n", threshold_date);

    struct group {
        char l_returnflag;
        char l_linestatus;
        int64_t sum_qty;
        int64_t sum_base_price;
        int64_t sum_disc_price;
        int64_t sum_charge;
        int64_t avg_qty;
        int64_t avg_price;
        int64_t avg_disc;
        uint64_t count_order;
    };
    unordered_map<uint16_t, std::unique_ptr<group>> groupBy;

    auto& lineitem = db.lineitem;
    for (size_t i = 0; i < lineitem.l_returnflag.size(); ++i) {
        if (lineitem.l_shipdate[i].raw > threshold_date) continue;

        uint16_t k = static_cast<uint16_t>(lineitem.l_returnflag[i]) << 8;
        k |= lineitem.l_linestatus[i];

        group* groupPtr;
        auto it = groupBy.find(k);
        if (it != groupBy.end()) {
            groupPtr = it->second.get();
        } else {
            // create new group
            auto g = std::make_unique<group>();
            groupPtr = g.get();
            groupBy[k] = std::move(g);
            groupPtr->l_returnflag = lineitem.l_returnflag[i];
            groupPtr->l_linestatus = lineitem.l_linestatus[i];
        }

        auto l_extendedprice = lineitem.l_extendedprice[i].raw;
        auto l_discount = lineitem.l_discount[i].raw;
        auto l_quantity = lineitem.l_quantity[i].raw;
        groupPtr->sum_qty += l_quantity;
        groupPtr->sum_base_price += l_extendedprice;
        groupPtr->sum_disc_price += l_extendedprice * (100 - l_discount); // sum(l_extendedprice * (1 - l_discount))
        groupPtr->sum_charge += l_extendedprice * (100 - l_discount) * (100 * lineitem.l_tax[i].raw); // sum(l_extendedprice * (1 - l_discount) * (1 + l_tax))
        groupPtr->avg_qty += l_quantity;
        groupPtr->avg_price += l_extendedprice;
        groupPtr->avg_disc += l_discount;
        groupPtr->count_order += 1;
    }

    // compute averages
    for (auto& t : groupBy) {
        int64_t cnt = t.second->count_order;
        // TODO adjust decimal point
        t.second->avg_qty /= cnt;
        t.second->avg_price /= cnt;
        t.second->avg_disc /= cnt;
    }

    std::vector<group*> sorted;
    sorted.reserve(groupBy.size());
    for (auto& t : groupBy) {
        sorted.push_back(t.second.get());
    }
    std::sort(sorted.begin(), sorted.end(), [](group* a, group* b) {
        return a->l_returnflag < b->l_returnflag || (a->l_returnflag == b->l_returnflag && a->l_linestatus < b->l_linestatus);
    });

    size_t tupleCount = 0;
    for (size_t i = 0; i < sorted.size(); i++) {
        //printf("%p\n", sorted[i]);
        auto& t = *sorted[i];
        cout << t.l_returnflag << "\t" << t.l_linestatus << "\t" << t.count_order << endl;
        tupleCount += t.count_order;
    }
    //printf("tupleCount: %lu\n", tupleCount);
}

/*
-- TPC-H Query 4

select
        o_orderpriority,
        count(*) as order_count
from
        orders
where
        o_orderdate >= date '1993-07-01'
        and o_orderdate < date '1993-10-01'
        and exists (
                select
                        *
                from
                        lineitem
                where
                        l_orderkey = o_orderkey
                        and l_commitdate < l_receiptdate
        )
group by
        o_orderpriority
order by
        o_orderpriority
*/
void query_4(Database& db) {
}


/*
-- TPC-H Query 14

select
        100.00 * sum(case
                when p_type like 'PROMO%'
                        then l_extendedprice * (1 - l_discount)
                else 0
        end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
        lineitem,
        part
where
        l_partkey = p_partkey
        and l_shipdate >= date '1995-09-01'
        and l_shipdate < date '1995-10-01'

*/
void query_14_part_build(Database& db) {
    int64_t sum1 = 0;
    int64_t sum2 = 0;

    constexpr std::string_view prefix = "PROMO";
    constexpr auto lower_shipdate = to_julian_day(1, 9, 1995); // 1995-09-01
    constexpr auto upper_shipdate = to_julian_day(1, 10, 1995); // 1995-10-01

    auto& part = db.part;
    auto& lineitem = db.lineitem;

    std::unordered_map<uint32_t, size_t> ht(part.p_partkey.size());
    for (size_t i = 0; i < part.p_partkey.size(); ++i) {
        ht.emplace(part.p_partkey[i], i);
    }

    // aggregation loop
    for (size_t i = 0; i < lineitem.l_partkey.size(); ++i) {
        if (lineitem.l_shipdate[i].raw < lower_shipdate ||
            lineitem.l_shipdate[i].raw >= upper_shipdate) {
            continue;
        }

        // probe
        auto it = ht.find(lineitem.l_partkey[i]);
        if (it == ht.end()) {
            continue;
        }
        size_t j = it->second;

        auto extendedprice = lineitem.l_extendedprice[i].raw;
        auto discount = lineitem.l_discount[i].raw;
        auto summand = extendedprice * (100 - discount);
        sum2 += summand;

        auto& type = part.p_type[j];
        if (std::strncmp(type.data(), prefix.data(), prefix.size()) == 0) {
            sum1 += summand;
        }
    }

    sum1 *= 1'000;
    sum2 /= 1'000;
    int64_t result = 100*sum1/sum2;
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);
}

void query_14_lineitem_build(Database& db) {
    int64_t sum1 = 0;
    int64_t sum2 = 0;

    constexpr std::string_view prefix = "PROMO";
    constexpr auto lower_shipdate = to_julian_day(1, 9, 1995); // 1995-09-01
    constexpr auto upper_shipdate = to_julian_day(1, 10, 1995); // 1995-10-01
/*
    cout << "lower: " << lower_shipdate << endl;
    cout << "upper: " << upper_shipdate << endl;
*/
    auto& part = db.part;
    auto& lineitem = db.lineitem;
//uint32_t min_partkey = 0xffffffff, max_partkey = 0;
    std::unordered_multimap<uint32_t, size_t> ht(part.p_partkey.size());// lineitem.l_partkey.size());
    for (size_t i = 0; i < lineitem.l_partkey.size(); ++i) {
        if (lineitem.l_shipdate[i].raw < lower_shipdate ||
            lineitem.l_shipdate[i].raw >= upper_shipdate) {
            continue;
        }
        ht.emplace(lineitem.l_partkey[i], i);/*
        max_partkey = std::max(max_partkey, lineitem.l_partkey[i]);
        min_partkey = std::min(min_partkey, lineitem.l_partkey[i]);*/
    }
//std::cout << "min_partkey: " << min_partkey << " max_partkey: " << max_partkey << std::endl;
    // aggregation loop
    for (size_t i = 0; i < part.p_partkey.size(); ++i) {
        // probe
        auto range = ht.equal_range(part.p_partkey[i]);
        for (auto it = range.first; it != range.second; ++it) {
            size_t j = it->second;

            auto extendedprice = lineitem.l_extendedprice[j].raw;
            auto discount = lineitem.l_discount[j].raw;
            auto summand = extendedprice * (100 - discount);
            sum2 += summand;

            auto& type = part.p_type[i];
            if (std::strncmp(type.data(), prefix.data(), prefix.size()) == 0) {
                sum1 += summand;
            }
        }
    }

    sum1 *= 1'000;
    sum2 /= 1'000;
    int64_t result = 100*sum1/sum2;
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);
}
