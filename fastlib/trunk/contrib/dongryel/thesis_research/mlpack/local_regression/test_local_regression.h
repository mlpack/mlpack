/** @file test_local_regression.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_TEST_LOCAL_REGRESSION_H
#define MLPACK_LOCAL_REGRESSION_TEST_LOCAL_REGRESSION_H

#include <boost/scoped_array.hpp>
#include <boost/test/unit_test.hpp>
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/parallel/random_dataset_generator.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include "mlpack/local_regression/local_regression_dev.h"
#include "mlpack/local_regression/local_regression_argument_parser.h"

namespace mlpack {
namespace local_regression {
namespace test_local_regression {
extern int num_dimensions_;
extern int num_points_;
}

class TestLocalRegression {

  private:

    template<typename QueryResultType>
    bool CheckAccuracy_(
      const QueryResultType &query_results,
      const std::vector<double> &naive_query_results,
      double relative_error) {

      // Compute the collective L1 norm of the products.
      double achieved_error = 0;
      for(unsigned int j = 0; j < naive_query_results.size(); j++) {
        double per_relative_error =
          fabs(naive_query_results[j] - query_results[j]) /
          fabs(naive_query_results[j]);
        achieved_error = std::max(achieved_error, per_relative_error);
        if(1.0 < per_relative_error) {
          std::cout << query_results[j] << " against " <<
                    naive_query_results[j] << ": " <<
                    per_relative_error << "\n";
        }
      }
      std::cout <<
                "Achieved a relative error of " << achieved_error << "\n";
      return achieved_error <= relative_error;
    }

  public:

    template<typename MetricType, typename TableType, typename KernelType>
    static void UltraNaive(
      int order,
      const MetricType &metric_in,
      TableType &query_table, TableType &reference_table,
      const KernelType &kernel,
      std::vector<double> &ultra_naive_query_results) {

      ultra_naive_query_results.resize(query_table.n_entries());

      int problem_dimension = (order == 0) ? 1 : query_table.n_attributes() + 1;
      for(int i = 0; i < query_table.n_entries(); i++) {
        core::table::DensePoint query_point;
        query_table.get(i, &query_point);

        // Monte Carlo result.
        core::monte_carlo::MeanVariancePairVector numerator;
        core::monte_carlo::MeanVariancePairMatrix denominator ;
        numerator.Init(problem_dimension);
        denominator.Init(problem_dimension, problem_dimension);

        for(int j = 0; j < reference_table.n_entries(); j++) {
          core::table::DensePoint reference_point;
          double reference_weight;
          reference_table.get(j, &reference_point, &reference_weight);

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

          // Accumulate the sum.
          numerator[0].push_back(reference_weight * kernel_value);
          denominator.get(0, 0).push_back(kernel_value);
          for(int n = 1; n < problem_dimension; n++) {
            numerator[n].push_back(
              reference_weight * kernel_value * reference_point[n - 1]);
            denominator.get(0, n).push_back(
              kernel_value * reference_point[n - 1]);
            denominator.get(n, 0).push_back(
              kernel_value * reference_point[n - 1]);
            for(int m = 1; m < problem_dimension; m++) {
              denominator.get(m, n).push_back(
                kernel_value * reference_point[m - 1] *
                reference_point[n - 1]);
            }
          }
        } // end of looping over each reference point.

        // Eigendecompose and solve the least squares problem.
        arma::mat left_hand_side;
        arma::vec right_hand_side;
        denominator.sample_means(&left_hand_side);
        numerator.sample_means(&right_hand_side);

        // Solve the linear system.
        arma::mat tmp_eigenvectors;
        arma::vec tmp_eigenvalues;
        arma::vec tmp_solution;
        arma::eig_sym(tmp_eigenvalues, tmp_eigenvectors, left_hand_side);
        tmp_solution.zeros(left_hand_side.n_rows);
        for(unsigned int m = 0; m < tmp_eigenvalues.n_elem; m++) {
          double dot_product = arma::dot(
                                 tmp_eigenvectors.col(m),
                                 right_hand_side);
          if(tmp_eigenvalues[m] > 1e-6) {
            tmp_solution +=
              (dot_product / tmp_eigenvalues[m]) * tmp_eigenvectors.col(m);
          }
        }

        // Take the dot product with the solution vector to get the
        // regression estimate.
        ultra_naive_query_results[i] = tmp_solution[0];
        for(unsigned int m = 1; m < tmp_solution.n_elem; m++) {
          ultra_naive_query_results[i] += tmp_solution[m] *
                                          query_point[m - 1];
        }
      } // end of looping over query point.
    }

    int StressTestMain() {
      for(int order = 0; order >= 0; order--) {
        for(int i = 0; i < 20; i++) {
          for(int k = 0; k < 4; k++) {
            // Randomly choose the number of dimensions and the points.
            mlpack::local_regression::test_local_regression::num_dimensions_ =
              core::math::RandInt(2, 5);
            mlpack::local_regression::test_local_regression::num_points_ =
              core::math::RandInt(500, 1001);

            switch(k) {
              case 0:
                StressTest <
                core::metric_kernels::GaussianKernel,
                     core::metric_kernels::LMetric<2> > (order);
                break;
              case 1:
                StressTest <
                core::metric_kernels::EpanKernel,
                     core::metric_kernels::LMetric<2> > (order);
                break;
              case 2:
                StressTest <
                core::metric_kernels::GaussianKernel,
                     core::metric_kernels::WeightedLMetric<2> > (order);
                break;
              case 3:
                StressTest <
                core::metric_kernels::EpanKernel,
                     core::metric_kernels::WeightedLMetric<2> > (order);
                break;
            }
          }
        }
      }
      return 0;
    }

    template<typename KernelType, typename MetricType>
    int StressTest(int order) {

      typedef core::table::Table <
      core::tree::GenMetricTree <
      mlpack::local_regression::LocalRegressionStatistic > > TableType;

      // The list of arguments.
      std::vector< std::string > args;

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Push in the reference target name.
      std::string reference_targets_in("random_targets.csv");
      args.push_back(
        std::string("--reference_targets_in=") + reference_targets_in);

      // Push in the prediction output file name.
      args.push_back(std::string("--predictions_out=predictions.txt"));

      // Push in the prescale option.
      args.push_back(std::string("--prescale=none"));

      // Push in the kernel type.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of dimensions: " <<
                mlpack::local_regression::test_local_regression::num_dimensions_ << "\n";
      std::cout << "Number of points: " <<
                mlpack::local_regression::test_local_regression::num_points_ << "\n";

      KernelType dummy_kernel;
      if(dummy_kernel.name() == "epan") {
        args.push_back(std::string("--kernel=epan"));
      }
      else if(dummy_kernel.name() == "gaussian") {
        args.push_back(std::string("--kernel=gaussian"));
      }

      // Push in the metric type.
      MetricType dummy_metric;
      if(dummy_metric.name() == "lmetric") {
        args.push_back(std::string("--metric=lmetric"));
      }
      else if(dummy_metric.name() == "weighted_lmetric") {
        args.push_back(std::string("--metric=weighted_lmetric"));

        // In this case, we need to generate the random scaling
        // factors.
        TableType random_scales_table;
        std::string random_scales_file_name("random_scales.csv");
        core::parallel::RandomDatasetGenerator::Generate(
          1, mlpack::local_regression::test_local_regression::num_dimensions_,
          0, std::string("none"), &random_scales_table);
        random_scales_table.Save(random_scales_file_name);
        args.push_back(
          std::string("--metric_scales_in=") + random_scales_file_name);
      }

      // Push in the leaf size.
      int leaf_size = core::math::RandInt(20, 30);
      std::stringstream leaf_size_sstr;
      leaf_size_sstr << "--leaf_size=" << leaf_size;
      args.push_back(leaf_size_sstr.str());

      // Push in the relative error argument.
      double relative_error = 0.1;
      std::stringstream relative_error_sstr;
      relative_error_sstr << "--relative_error=" << relative_error;
      args.push_back(relative_error_sstr.str());

      // Push in the order argument.
      std::stringstream order_sstr;
      order_sstr << "--order=" << order;
      args.push_back(order_sstr.str());

      // Push in the randomly generated bandwidth.
      double bandwidth =
        core::math::Random(
          0.05 * sqrt(
            mlpack::local_regression::test_local_regression::num_dimensions_),
          0.1 * sqrt(
            mlpack::local_regression::test_local_regression::num_dimensions_));
      std::stringstream bandwidth_sstr;
      bandwidth_sstr << "--bandwidth=" << bandwidth;
      args.push_back(bandwidth_sstr.str());

      // Generate the random dataset and save it.
      TableType random_table;
      core::parallel::RandomDatasetGenerator::Generate(
        mlpack::local_regression::test_local_regression::num_dimensions_,
        mlpack::local_regression::test_local_regression::num_points_, 0,
        std::string("none"), &random_table);
      random_table.Save(references_in, &reference_targets_in);

      // Parse the local regression arguments.
      mlpack::local_regression::LocalRegressionArguments <
      TableType, MetricType >
      local_regression_arguments;
      boost::program_options::variables_map vm;
      mlpack::local_regression::LocalRegressionArgumentParser::ConstructBoostVariableMap(args, &vm);
      mlpack::local_regression::LocalRegressionArgumentParser::ParseArguments(
        vm, &local_regression_arguments);

      std::cout << "Bandwidth value " << bandwidth << "\n";

      // Call the local regression driver.
      mlpack::local_regression::LocalRegression <
      TableType, KernelType, MetricType > local_regression_instance;
      local_regression_instance.Init(
        local_regression_arguments,
        (typename mlpack::local_regression::LocalRegression <
         TableType, KernelType, MetricType >::GlobalType *) NULL);

      // Compute the result.
      mlpack::local_regression::LocalRegressionResult local_regression_result;
      local_regression_instance.Compute(
        local_regression_arguments, &local_regression_result);

      // Call the ultra-naive.
      std::vector<double> ultra_naive_local_regression_result;

      UltraNaive(
        order,
        local_regression_arguments.metric_,
        *(local_regression_arguments.reference_table_),
        *(local_regression_arguments.reference_table_),
        local_regression_instance.global().kernel(),
        ultra_naive_local_regression_result);
      if(CheckAccuracy_(
            local_regression_result.regression_estimates_,
            ultra_naive_local_regression_result,
            local_regression_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
      }

      return 0;
    };
};
}
}

#endif
