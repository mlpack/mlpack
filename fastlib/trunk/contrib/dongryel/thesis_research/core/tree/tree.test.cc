/** @file tree.test.cc
 *
 *  A "stress" test driver for trees.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "core/metric_kernels/lmetric.h"
#include "core/table/table.h"
#include <time.h>

namespace core {
namespace tree {

template<typename TableType>
class TestTree {

  private:

    bool TestTreeIterator_(
      typename TableType::TreeType *node,
      TableType &table) {

      typename TableType::TreeIterator node_it =
        table.get_node_iterator(node);
      do {
        core::table::DensePoint point;
        int point_id;
        node_it.Next(&point, &point_id);
        core::table::DensePoint compare_point;
        table.get(point_id, &compare_point);

        for(int i = 0; i < point.length(); i++) {
          if(point[i] != compare_point[i]) {
            return false;
          }
        }
      }
      while(node_it.HasNext());

      if(table.node_is_leaf(node) == false) {
        return TestTreeIterator_(
                 table.get_node_left_child(node), table) &&
               TestTreeIterator_(
                 table.get_node_right_child(node), table);
      }
      return true;
    }

    void GenerateRandomDataset_(
      int num_dimensions,
      int num_points,
      TableType *random_dataset) {

      random_dataset->Init(num_dimensions, num_points);

      for(int j = 0; j < num_points; j++) {
        core::table::DensePoint point;
        random_dataset->get(j, &point);
        for(int i = 0; i < num_dimensions; i++) {
          point[i] = core::math::Random(0.1, 1.0);
        }
      }
    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 10; i++) {
        int num_dimensions = core::math::RandInt(3, 20);
        int num_points = core::math::RandInt(3000, 5001);
        if(StressTest(num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      return 0;
    }

    bool StressTest(int num_dimensions, int num_points) {

      std::vector< std::string > args;

      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Generate the random dataset and save it.
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      random_table.Save(references_in);

      // Reload the table twice and build the tree on one of them.
      TableType reordered_table;
      reordered_table.Init(references_in);
      TableType original_table;
      original_table.Init(references_in);
      core::metric_kernels::LMetric<2> l2_metric;
      reordered_table.IndexData(l2_metric, 20);
      for(int i = 0; i < reordered_table.n_entries(); i++) {
        core::table::DensePoint reordered_point;
        core::table::DensePoint original_point;
        reordered_table.get(i, &reordered_point);
        original_table.get(i, &original_point);
        for(int j = 0; j < reordered_table.n_attributes(); j++) {
          if(reordered_point[j] != original_point[j]) {
            return false;
          }
        }
      }

      // Now test the node iterator at each level of the tree.
      return TestTreeIterator_(
               reordered_table.get_tree(), reordered_table);
    }
};
};
};

BOOST_AUTO_TEST_SUITE(TestSuiteKde)
BOOST_AUTO_TEST_CASE(TestCaseKde) {
  srand(time(NULL));

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> > TableType;

  // Call the tests.
  core::tree::TestTree<TableType> tree_test;
  tree_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
