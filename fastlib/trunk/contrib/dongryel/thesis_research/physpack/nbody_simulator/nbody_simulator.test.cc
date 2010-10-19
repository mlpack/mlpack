/** @file three_body.test.cc
 *
 *  A "stress" test driver for three-body potentials.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "nbody_simulator_dev.h"
#include <time.h>

namespace nbody_simulator {
namespace test_nbody_simulator {
int num_points_;
};

class TestNbodySimulator {

  private:

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
      core::table::Table *random_dataset) {

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
      core::table::Table &table,
      const physpack::nbody_simulator::AxilrodTeller &potential,
      std::vector<double> &ultra_naive_query_results) {

      // Loop over distinct 3-tuples.
      core::gnp::TripleDistanceSq triple_distance_sq;
      for(int i = 0; i < table.n_entries() - 2; i++) {
        core::table::DenseConstPoint i_th_point;
        table.get(i, &i_th_point);
        triple_distance_sq.ReplaceOnePoint(metric_in, i_th_point, 0);

        for(int j = i + 1; j < table.n_entries() - 1; j++) {
          core::table::DenseConstPoint j_th_point;
          table.get(j, &j_th_point);
          triple_distance_sq.ReplaceOnePoint(metric_in, j_th_point, 1);

          for(int k = j + 1; k < table.n_entries(); k++) {
            core::table::DenseConstPoint k_th_point;
            table.get(k, &k_th_point);
            triple_distance_sq.ReplaceOnePoint(metric_in, k_th_point, 2);

            // Compute the potential induced by (i, j, k) and add it
            // to each of i_th, j_th, and k_th point.
            double potential_induced = potential.EvalUnnormOnSq(
                                         triple_distance_sq);
            ultra_naive_query_results[i] += potential_induced;
            ultra_naive_query_results[j] += potential_induced;
            ultra_naive_query_results[k] += potential_induced;
          }
        }
      }
    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 20; i++) {
        for(int k = 0; k < 2; k++) {
          // Randomly choose the number of dimensions and the points.
          ml::kde::test_kde::num_dimensions_ = core::math::RandInt(3, 20);
          ml::kde::test_kde::num_points_ = core::math::RandInt(500, 1001);
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
                ml::kde::test_kde::num_dimensions_ << "\n";
      std::cout << "Number of points: " <<
                ml::kde::test_kde::num_points_ << "\n";

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
          0.1 * sqrt(2.0 * ml::kde::test_kde::num_dimensions_),
          0.2 * sqrt(2.0 * ml::kde::test_kde::num_dimensions_));
      std::stringstream bandwidth_sstr;
      bandwidth_sstr << "--bandwidth=" << bandwidth;
      args.push_back(bandwidth_sstr.str());

      // Generate the random dataset and save it.
      core::table::Table random_table;
      GenerateRandomDataset_(
        ml::kde::test_kde::num_dimensions_,
        ml::kde::test_kde::num_points_, &random_table);
      random_table.Save(references_in);

      // Parse the nbody simulator arguments.
      ml::NbodySimulatorArguments nbody_simulator_arguments;
      ml::NbodySimulator::ParseArguments(args, &nbody_simulator_arguments);

      std::cout << "Bandwidth value " << bandwidth << "\n";

      // Call the nbody simulator driver.
      ml::NbodySimulator nbody_simulator_instance;
      nbody_simulator_instance.Init(nbody_simulator_arguments);

      // Compute the result.
      ml::NbodySimulatorResult< std::vector<double> > nbody_simulator_result;
      nbody_simulator_instance.Compute(
        nbody_simulator_arguments, &nbody_simulator_result);

      // Call the ultra-naive.
      std::vector<double> ultra_naive_nbody_simulator_result(
        nbody_simulator_arguments.reference_table_->n_entries(), 0.0);

      UltraNaive_(
        *(nbody_simulator_arguments.metric_),
        *(nbody_simulator_arguments.reference_table_),
        nbody_simulator_instance.global().potential(),
        ultra_naive_nbody_simulator_result);
      if(CheckAccuracy_(
            nbody_simulator_result.potentials_,
            ultra_naive_nbody_simulator_result,
            nbody_simulator_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
      }

      return 0;
    };
};
};
};

BOOST_AUTO_TEST_SUITE(TestSuiteNbodySimulator)
BOOST_AUTO_TEST_CASE(TestCaseNbodySimulator) {
  srand(time(NULL));

  // Call the tests.
  ml::kde::TestNbodySimulator nbody_simulator_test;
  nbody_simulator_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
