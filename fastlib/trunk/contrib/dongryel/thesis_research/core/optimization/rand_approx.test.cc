/** @file rand_approx.test.cc
 *
 *  Tests various random approximation routines.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <time.h>
#include "core/optimization/rand_range_finder.h"
#include "core/table/dense_matrix.h"
#include "core/table/table.h"
#include "core/tree/gen_metric_tree.h"

namespace core {
namespace optimization {
template<typename TableType>
class TestRandApprox {

  private:

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

        // Set the weight to the random one.
        random_dataset->weights().set(
          0, j, core::math::Random(1.0, 5.0));
      }
    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 10; i++) {
        int num_dimensions = core::math::RandInt(3, 20);
        int num_points = core::math::RandInt(1300, 2001);
        if(StressTest(num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      return 0;
    }

    int StressTest(int num_dimensions, int num_points) {

      // Generate the random dataset and save it.
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      arma::mat random_matrix;
      core::table::DenseMatrixToArmaMat(
        random_table.data(), &random_matrix);

      arma::mat basis;
      core::optimization::RandRangeFinder::Compute(
        random_matrix, 0.01, 0.9, &basis);

      return 1;
    }
};
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteRandApprox)
BOOST_AUTO_TEST_CASE(TestCaseRandApprox) {

  // Table type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> >
  GenMetricTreeTableType;

  // Call the test.
  core::optimization::TestRandApprox< GenMetricTreeTableType > test;
  test.StressTestMain();
}
BOOST_AUTO_TEST_SUITE_END()
