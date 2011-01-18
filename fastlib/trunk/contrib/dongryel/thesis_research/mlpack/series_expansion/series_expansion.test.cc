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
#include "mlpack/series_expansion/hypercube_farfield_dev.h"
#include "mlpack/series_expansion/hypercube_local_dev.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/multivariate_local_dev.h"

namespace mlpack {
namespace series_expansion {

template <
typename TableType,
         enum mlpack::series_expansion::CartesianExpansionType ExpansionType >
class SeriesExpansionTest {

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
      }
    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 10; i++) {
        int num_dimensions = core::math::RandInt(3, 5);
        int num_points = core::math::RandInt(30, 100);
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
      int max_order = 6 - num_dimensions;
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);

      // Form a Cartesian expansion global object.
      mlpack::series_expansion::CartesianExpansionGlobal<ExpansionType> global;
      global.Init(max_order, random_table.n_attributes());
      global.Print();

      // Form a far-field expansion and evaluate.
      mlpack::series_expansion::CartesianFarField<ExpansionType> farfield;
      return true;
    }
};
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteSeriesExpansion)
BOOST_AUTO_TEST_CASE(TestCaseSeriesExpansion) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> > TableType;

  // Call the tests.
  mlpack::series_expansion::SeriesExpansionTest <
  TableType, mlpack::series_expansion::MULTIVARIATE > multivariate_test;
  multivariate_test.StressTestMain();
  mlpack::series_expansion::SeriesExpansionTest <
  TableType, mlpack::series_expansion::HYPERCUBE > hypercube_test;
  hypercube_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
