/** @file series_expansion.test.cc
 *
 *  The test driver for series expansion library.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <time.h>
#include "core/metric_kernels/lmetric.h"
#include "core/table/table.h"
#include "core/math/math_lib.h"
#include "mlpack/series_expansion/cartesian_expansion_global_dev.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/hypercube_farfield_dev.h"

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

      if(node->is_leaf() == false) {
        return TestTreeIterator_(node->left(), table) &&
               TestTreeIterator_(node->right(), table);
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

      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

      // Generate a random table.
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      core::metric_kernels::LMetric<2> l2_metric;
      random_table.IndexData(l2_metric, 20);

      return true;
    }
};
};
};

BOOST_AUTO_TEST_SUITE(TestSuiteKde)
BOOST_AUTO_TEST_CASE(TestCaseKde) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> > TableType;

  // Call the tests.
  core::tree::TestTree<TableType> tree_test;
  tree_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
