/** @file three_body.test.cc
 *
 *  A "stress" test driver for three-body potentials.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "core/math/math_lib.h"
#include "core/tree/gen_metric_tree.h"
#include "physpack/nbody_simulator/nbody_simulator_dev.h"
#include <time.h>

namespace physpack {
namespace nbody_simulator {
namespace test_nbody_simulator {
int num_points_;
};

class TestNbodySimulator {

  private:

    typedef core::table::Table <
    core::tree::GenMetricTree <
    physpack::nbody_simulator::NbodySimulatorStatistic > > TableType;

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
                    per_relative_error << " for " << j << "-th.\n";
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

    template<typename MetricType>
    void UltraNaive_(
      const MetricType &metric_in,
      TableType &table,
      const physpack::nbody_simulator::AxilrodTeller &potential,
      std::vector<double> &ultra_naive_query_results) {

      // Loop over distinct 3-tuples.
      core::gnp::TripleDistanceSq triple_distance_sq;
      for(int i = 0; i < table.n_entries() - 2; i++) {
        core::table::DensePoint i_th_point;
        table.get(i, &i_th_point);
        triple_distance_sq.ReplaceOnePoint(metric_in, i_th_point, i, 0);

        for(int j = i + 1; j < table.n_entries() - 1; j++) {
          core::table::DensePoint j_th_point;
          table.get(j, &j_th_point);
          triple_distance_sq.ReplaceOnePoint(metric_in, j_th_point, j, 1);

          for(int k = j + 1; k < table.n_entries(); k++) {
            core::table::DensePoint k_th_point;
            table.get(k, &k_th_point);
            triple_distance_sq.ReplaceOnePoint(metric_in, k_th_point, k, 2);

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
        // Randomly choose the number of points.
        nbody_simulator::test_nbody_simulator::num_points_ =
          core::math::RandInt(100, 200);
        StressTest();
      }
      return 0;
    }

    int StressTest() {

      std::vector< std::string > args;

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Push in the densities output file name.
      args.push_back(std::string("--potentials_out=potentials.txt"));

      // Push in the kernel type.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of points: " <<
                nbody_simulator::test_nbody_simulator::num_points_ << "\n";

      // Push in the leaf size.
      int leaf_size = 20;
      std::stringstream leaf_size_sstr;
      leaf_size_sstr << "--leaf_size=" << leaf_size;
      args.push_back(leaf_size_sstr.str());

      // Push in the relative error.
      std::stringstream relative_error_sstr;
      relative_error_sstr << "--relative_error=" << 0.1;
      args.push_back(relative_error_sstr.str());

      // Generate the random dataset and save it.
      TableType random_table;
      GenerateRandomDataset_(
        3, nbody_simulator::test_nbody_simulator::num_points_, &random_table);
      random_table.Save(references_in);

      // Parse the nbody simulator arguments.
      physpack::nbody_simulator::NbodySimulatorArguments<TableType>
      nbody_simulator_arguments;
      physpack::nbody_simulator::NbodySimulator<TableType>::ParseArguments(
        args, &nbody_simulator_arguments);

      // Call the nbody simulator driver.
      physpack::nbody_simulator::NbodySimulator<TableType>
      nbody_simulator_instance;
      nbody_simulator_instance.Init(nbody_simulator_arguments);

      // Compute the result.
      physpack::nbody_simulator::NbodySimulatorResult nbody_simulator_result;
      nbody_simulator_instance.Compute(
        nbody_simulator_arguments, &nbody_simulator_result);

      // First check whether the number of tuples encountered for each
      // point is correct.
      for(unsigned int i = 0; i < nbody_simulator_result.pruned_.size(); i++) {
        if(nbody_simulator_result.pruned_[i] !=
            nbody_simulator_instance.global().total_num_tuples()) {
          std::cerr << "The number of tuples for " << i << "-th point is " <<
                    nbody_simulator_result.pruned_[i] << " but it should be " <<
                    nbody_simulator_instance.global().total_num_tuples();
        }
      }

      // Call the ultra-naive.
      std::vector<double> ultra_naive_nbody_simulator_result(
        nbody_simulator_arguments.table_->n_entries(), 0.0);

      UltraNaive_(
        *(nbody_simulator_arguments.metric_),
        *(nbody_simulator_arguments.table_),
        nbody_simulator_instance.global().potential(),
        ultra_naive_nbody_simulator_result);
      if(CheckAccuracy_(
            nbody_simulator_result.potential_e_,
            ultra_naive_nbody_simulator_result,
            nbody_simulator_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
      }

      return 0;
    };
};
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteNbodySimulator)
BOOST_AUTO_TEST_CASE(TestCaseNbodySimulator) {

  // Call the tests.
  physpack::nbody_simulator::TestNbodySimulator nbody_simulator_test;
  nbody_simulator_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
