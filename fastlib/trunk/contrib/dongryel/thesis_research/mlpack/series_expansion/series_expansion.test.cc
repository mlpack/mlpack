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
#include "mlpack/series_expansion/kernel_aux.h"
#include "mlpack/series_expansion/hypercube_farfield_dev.h"
#include "mlpack/series_expansion/hypercube_local_dev.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/multivariate_local_dev.h"

namespace mlpack {
namespace series_expansion {

template <typename TableType, typename KernelAuxType >
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
        int num_dimensions = core::math::RandInt(2, 5);
        int num_points = core::math::RandInt(2000, 3000);
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
      int leaf_size = core::math::RandInt(10, 20);
      core::metric_kernels::LMetric<2> l2_metric;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      random_table.IndexData(l2_metric, leaf_size);
      std::vector<typename TableType::TreeType *> leaf_nodes;
      random_table.get_leaf_nodes(random_table.get_tree(), &leaf_nodes);

      // Randomly select a reference region and a query region.
      typename TableType::TreeType *reference_node =
        leaf_nodes[core::math::RandInt(leaf_nodes.size())];
      typename TableType::TreeType *query_node =
        leaf_nodes[core::math::RandInt(leaf_nodes.size())];
      core::math::Range squared_distance_range =
        reference_node->bound().RangeDistanceSq(
          l2_metric, query_node->bound());

      // Form a Cartesian expansion global object.
      KernelAuxType kernel_aux;
      double bandwidth = core::math::Random(
                           0.13 * num_dimensions, 0.2 * num_dimensions);
      kernel_aux.Init(bandwidth, max_order, random_table.n_attributes());
      kernel_aux.global().CheckIntegrity();

      // Form a far-field expansion and evaluate.
      mlpack::series_expansion::CartesianFarField <
      KernelAuxType::ExpansionType > farfield;

      // It is important that the weight vector matches the number of
      // points in the reference set.
      core::table::DensePoint weights;
      weights.Init(random_table.n_entries());
      for(int i = 0; i < weights.length(); i++) {
        weights[i] = core::math::Random(0.25, 1.25);
      }
      farfield.Init(kernel_aux, reference_node->bound().center());

      // Determine the truncation order and form the farfield
      // expansion up to that order. Compare the evaluation using the
      // naive method.
      double evaluation_max_error =
        core::math::Random(0.025, 0.003) * reference_node->count();
      double farfield_actual_error = 0;
      int farfield_truncation_order =
        kernel_aux.OrderForEvaluatingFarField(
          reference_node->bound(), query_node->bound(),
          squared_distance_range.lo, squared_distance_range.hi,
          evaluation_max_error, &farfield_actual_error);
      printf("Far field truncation order: %d\n", farfield_truncation_order);
      if(farfield_truncation_order >= 0) {
        farfield.AccumulateCoeffs(
          kernel_aux, random_table.data(),
          weights, 0, random_table.n_entries(), farfield_truncation_order);
      }

      // Now form a local expansion of the reference node onto the
      // query node.
      mlpack::series_expansion::CartesianLocal <
      KernelAuxType::ExpansionType > local;
      local.Init(kernel_aux, query_node->bound().center());
      double local_actual_error = 0;
      int local_truncation_order =
        kernel_aux.OrderForEvaluatingLocal(
          reference_node->bound(), query_node->bound(),
          squared_distance_range.lo, squared_distance_range.hi,
          evaluation_max_error, &local_actual_error);
      printf("Local expansion truncation order: %d\n", local_truncation_order);
      if(local_truncation_order >= 0) {
        local.AccumulateCoeffs(
          kernel_aux, random_table.data(),
          weights, 0, random_table.n_entries(), local_truncation_order);
      }
      printf("\n");

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
  TableType, mlpack::series_expansion::GaussianKernelAux > gaussian_kernel_test;
  gaussian_kernel_test.StressTestMain();
  std::cout << "Passed the Gaussian kernel $O(D^p)$ test.\n";

  mlpack::series_expansion::SeriesExpansionTest <
  TableType, mlpack::series_expansion::GaussianKernelMultAux >
  gaussian_mult_kernel_test;
  gaussian_mult_kernel_test.StressTestMain();
  std::cout << "Passed the Gaussian kernel $O(p^D)$ test.\n";

  mlpack::series_expansion::SeriesExpansionTest <
  TableType, mlpack::series_expansion::EpanKernelAux > epan_kernel_test;
  epan_kernel_test.StressTestMain();
  std::cout << "Passed the Epanechnikov kernel $O(D^p)$ test.\n";

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
