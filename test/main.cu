#include <sstream>
#include <vector>
#include <string>
#include <iostream>

#include "harmonia.cuh"

#include "utils.hpp"

#include "gtest/gtest.h"

/*
namespace my {
namespace project {
namespace {

// The fixture for testing class Foo.
class FooTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if their bodies would
  // be empty.

  FooTest() {
     // You can do set-up work for each test here.
  }

  ~FooTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

// Tests that the Foo::Bar() method does Abc.
TEST_F(FooTest, MethodBarDoesAbc) {
  const std::string input_filepath = "this/package/testdata/myinputfile.dat";
  const std::string output_filepath = "this/package/testdata/myoutputfile.dat";
  Foo f;
  EXPECT_EQ(f.Bar(input_filepath, output_filepath), 0);
}

// Tests that Foo does Xyz.
TEST_F(FooTest, DoesXyz) {
  // Exercises the Xyz feature of Foo.
}

}  // namespace
}  // namespace project
}  // namespace my
*/

using namespace harmonia;

namespace testing {

template<class T> using harmonia_allocator = std::allocator<T>;

using harmonia_type = harmonia_tree<uint32_t, uint32_t, harmonia_allocator, 4 + 1, std::numeric_limits<uint32_t>::max()>;

// Fixture for testing harmonia
struct HarmoniaTest : public ::testing::Test {

    harmonia_type tree;

    HarmoniaTest() {
        
    }
};

using ::tmpl_to_string;


template<class VectorType, class TreeType>
void dump_levels(const VectorType& levels, const TreeType& tree) {
    const auto max_keys = TreeType::get_max_keys();
    size_t level_num = 0;
    for (const auto& level : levels) {
        std::cout << "level " << level_num << ": " << tmpl_to_string(level) << "\n";

        std::cout << "nodes:" << "\n";

        for (size_t node_idx = 0; node_idx < level.node_count; ++node_idx) {
            const auto keys_start = max_keys * (level.node_count_prefix_sum + node_idx);
            std::cout << "node " << node_idx << ": " << stringify(tree.keys.begin() + keys_start, tree.keys.begin() + keys_start + max_keys) << "\n";
        }

        std::cout << std::endl;

        level_num += 1;
    }
}

/*
// Tests that Foo does Xyz.
TEST_F(HarmoniaTest, GatherTreeInfo) {
    std::vector<uint32_t> keys(21);
    std::iota(keys.begin(), keys.end(), 0);

    //ASSERT_EQ(harmonia_type::max_keys, 4);
    ASSERT_EQ(harmonia_type::get_max_keys(), 4);

    auto levels = tree.gather_tree_info(keys);

//    tree.create_node_descriptions(levels);

    //auto r = stringify(levels.begin(), levels.end());
    //std::cout << r << std::endl;
    std::cout << stringify(keys.begin(), keys.end()) << std::endl;

    //dump_levels(levels, tree);
}

TEST_F(HarmoniaTest, ConstructTree) {
    std::vector<uint32_t> keys(21);
    std::iota(keys.begin(), keys.end(), 0);

    //ASSERT_EQ(harmonia_type::max_keys, 4);
    ASSERT_EQ(harmonia_type::get_max_keys(), 4);

    tree.construct(keys);

    std::cout << "keys: " << stringify(tree.keys.begin(), tree.keys.end()) << std::endl;
    std::cout << "children: " << stringify(tree.children.begin(), tree.children.end()) << std::endl;

    const auto levels = tree.gather_tree_info(keys);
    dump_levels(levels, tree);
}
*/

TEST_F(HarmoniaTest, Lookup) {
    std::vector<uint32_t> keys(144);//87);
    std::iota(keys.begin(), keys.end(), 0);


    tree.construct(keys);

//    std::cout << "keys: " << stringify(tree.keys.begin(), tree.keys.end()) << std::endl;
//    std::cout << "children: " << stringify(tree.children.begin(), tree.children.end()) << std::endl;

#if 0
    const auto levels = tree.gather_tree_info(keys);
    dump_levels(levels, tree);
#endif

//    std::cout << "keys: " << stringify(tree.keys.begin(), tree.keys.end()) << std::endl;
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto tid = tree.lookup(keys[i]);
        EXPECT_EQ(tid, i);
    }
}

}

/*
template<>
std::string tmpl_to_string<testing::harmonia_type::node_description>(const testing::harmonia_type::node_description& node) {
    std::ostringstream s;
    s << "is_leaf: " << node.is_leaf << " count: " << node.count << " key_start_range: " << node.key_start_range
      << " child_ref_start: " << node.child_ref_start << "\n";
    return s.str();
}
*/
 
#if 0
template<>
std::string tmpl_to_string<testing::harmonia_type::level_data>(const testing::harmonia_type::level_data& level) {
    std::ostringstream s;
    s << "Key_count: " << level.key_count << " node_count: " << level.node_count << "\n"
      << " key_count_prefix_sum: " << level.key_count_prefix_sum << " node_count_prefix_sum: " << level.node_count_prefix_sum;
/*
    if (!level.nodes.empty()) {
        s << "nodes: " << stringify(level.nodes.begin(), level.nodes.end()) << "\n";
    }
*/
    return s.str();
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
