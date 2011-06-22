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
      for(int i = 0; i < 1; i++) {
        int num_dimensions = core::math::RandInt(30, 100);
        int num_points = core::math::RandInt(10, 21);
        if(StressTest(num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      return 0;
    }

    int StressTest(int num_dimensions, int num_points) {

      // Generate the random dataset.
      std::cerr << "Generating " << num_points << " points of " <<
                num_dimensions << " dimensionality.\n";
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      arma::mat random_matrix;
      core::table::DenseMatrixToArmaMat(
        random_table.data(), &random_matrix);

      // Compute.
      arma::mat basis;
      core::optimization::RandRangeFinder::Compute(
        random_matrix, 0.1, 0.9, &basis);

      // Compute the two norm error.
      arma::mat error = random_matrix;
      for(unsigned int j = 0; j < random_matrix.n_cols; j++) {
        for(unsigned int k = 0; k < basis.n_cols; k++) {
          double dot_product = arma::dot(error.col(j) , basis.col(k));
          error.col(j) -= dot_product * basis.col(k);
        }
      }
      std::cerr << "Epsilon achieved: " << arma::norm(error, 2) <<
                " achieved with " << basis.n_cols << " basis vectors.\n";

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
