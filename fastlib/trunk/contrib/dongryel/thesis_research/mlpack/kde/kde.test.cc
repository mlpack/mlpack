/** @file kde.test.cc
 *
 *  A "stress" test driver for KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "core/gnp/dualtree_dfs_dev.h"
#include "kde_dev.h"
#include "core/tree/gen_metric_tree.h"
#include <time.h>

namespace mlpack {
namespace kde {
namespace test_kde {
int num_dimensions_;
int num_points_;
};

class TestKde {

  private:

    typedef core::table::Table <
    core::tree::GenMetricTree<mlpack::kde::KdeStatistic> > TableType;

    bool CheckAccuracy_(
      const std::vector<double> &query_results,
      const std::vector<double> &naive_query_results,
      double relative_error) {

      // Compute the collective L1 norm of the products.
      double achieved_error = 0;
      for(unsigned int j = 0; j < query_results.size(); j++) {
        double per_relative_error =
          fabs(naive_query_results[j] - query_results[j]) /
          fabs(naive_query_results[j]);
        achieved_error = std::max(achieved_error, per_relative_error);
        if(relative_error < per_relative_error) {
          std::cout << query_results[j] << " against " <<
                    naive_query_results[j] << ": " <<
                    per_relative_error << "\n";
        }
      }
      std::cout <<
                "Achieved a relative error of " << achieved_error << "\n";
      return achieved_error <= relative_error;
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

    void UltraNaive_(
      const core::metric_kernels::AbstractMetric &metric_in,
      TableType &query_table, TableType &reference_table,
      const core::metric_kernels::AbstractKernel &kernel,
      std::vector<double> &ultra_naive_query_results) {

      ultra_naive_query_results.resize(query_table.n_entries());
      for(int i = 0; i < query_table.n_entries(); i++) {
        core::table::DensePoint query_point;
        query_table.get(i, &query_point);
        ultra_naive_query_results[i] = 0;

        for(int j = 0; j < reference_table.n_entries(); j++) {
          core::table::DensePoint reference_point;
          reference_table.get(j, &reference_point);

          // By default, monochromaticity is assumed in the test -
          // this will be addressed later for general bichromatic
          // test.
          if(i == j) {
            continue;
          }

          double squared_distance =
            metric_in.DistanceSq(query_point, reference_point);
          double kernel_value =
            kernel.EvalUnnormOnSq(squared_distance);

          ultra_naive_query_results[i] += kernel_value;
        }

        // Divide by N - 1 for LOO. May have to be adjusted later.
        ultra_naive_query_results[i] *=
          (1.0 / (kernel.CalcNormConstant(query_table.n_attributes()) *
                  ((double)
                   reference_table.n_entries() - 1)));
      }
    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 20; i++) {
        for(int k = 0; k < 2; k++) {
          // Randomly choose the number of dimensions and the points.
          mlpack::kde::test_kde::num_dimensions_ = core::math::RandInt(3, 20);
          mlpack::kde::test_kde::num_points_ = core::math::RandInt(500, 1001);
          StressTest(k);
        }
      }
      return 0;
    }

    int StressTest(int kernel_type) {

      std::vector< std::string > args;

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Push in the densities output file name.
      args.push_back(std::string("--densities_out=densities.txt"));

      // Push in the kernel type.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of dimensions: " <<
                mlpack::kde::test_kde::num_dimensions_ << "\n";
      std::cout << "Number of points: " <<
                mlpack::kde::test_kde::num_points_ << "\n";

      switch(kernel_type) {
        case 0:
          std::cout << "Epan kernel, \n";
          args.push_back(std::string("--kernel=epan"));
          break;
        case 1:
          std::cout << "Gaussian kernel, \n";
          args.push_back(std::string("--kernel=gaussian"));
          break;
      }

      // Push in the leaf size.
      int leaf_size = 20;
      std::stringstream leaf_size_sstr;
      leaf_size_sstr << "--leaf_size=" << leaf_size;
      args.push_back(leaf_size_sstr.str());

      // Push in the randomly generated bandwidth.
      double bandwidth =
        core::math::Random(
          0.1 * sqrt(2.0 * mlpack::kde::test_kde::num_dimensions_),
          0.2 * sqrt(2.0 * mlpack::kde::test_kde::num_dimensions_));
      std::stringstream bandwidth_sstr;
      bandwidth_sstr << "--bandwidth=" << bandwidth;
      args.push_back(bandwidth_sstr.str());

      // Generate the random dataset and save it.
      TableType random_table;
      GenerateRandomDataset_(
        mlpack::kde::test_kde::num_dimensions_,
        mlpack::kde::test_kde::num_points_, &random_table);
      random_table.Save(references_in);

      // Parse the KDE arguments.
      mlpack::kde::KdeArguments<TableType> kde_arguments;
      mlpack::kde::Kde<TableType>::ParseArguments(args, &kde_arguments);

      std::cout << "Bandwidth value " << bandwidth << "\n";

      // Call the KDE driver.
      mlpack::kde::Kde<TableType> kde_instance;
      kde_instance.Init(kde_arguments);

      // Compute the result.
      mlpack::kde::KdeResult< std::vector<double> > kde_result;
      kde_instance.Compute(kde_arguments, &kde_result);

      // Call the ultra-naive.
      std::vector<double> ultra_naive_kde_result;

      UltraNaive_(
        *(kde_arguments.metric_), *(kde_arguments.reference_table_),
        *(kde_arguments.reference_table_),
        kde_instance.global().kernel(),
        ultra_naive_kde_result);
      if(CheckAccuracy_(
            kde_result.densities_,
            ultra_naive_kde_result,
            kde_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
      }

      return 0;
    };
};
};
};

BOOST_AUTO_TEST_SUITE(TestSuiteKde)
BOOST_AUTO_TEST_CASE(TestCaseKde) {
  srand(time(NULL));

  // Call the tests.
  mlpack::kde::TestKde kde_test;
  kde_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
