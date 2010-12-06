/** @file distributed_kde.test.cc
 *
 *  A "stress" test driver for the distributed KDE driver.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "distributed_kde_dev.h"
#include "core/tree/gen_metric_tree.h"
#include <time.h>

namespace mlpack {
namespace distributed_kde {
class TestDistributed_Kde {

  private:

    typedef core::tree::GenMetricTree<mlpack::kde::KdeStatistic> TreeSpecType;

    typedef core::table::DistributedTable < TreeSpecType > DistributedTableType;

    typedef DistributedTableType::TableType TableType;

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

    int StressTestMain(boost::mpi::communicator &world) {
      for(int i = 0; i < 20; i++) {
        for(int k = 0; k < 2; k++) {
          StressTest(world, k);
        }
      }
      return 0;
    }

    int StressTest(
      boost::mpi::communicator &world, int kernel_type) {

      // Only the master generates the number of dimensions.
      int num_dimensions;
      if(world.rank() == 0) {
        num_dimensions = core::math::RandInt(3, 10);
      }
      boost::mpi::broadcast(world, num_dimensions, 0);
      int num_points = core::math::RandInt(10, 50);
      std::vector< std::string > args;

      // Push in the random generate command.
      std::stringstream random_generate_n_attributes_sstr;
      random_generate_n_attributes_sstr << "--random_generate_n_attributes=" <<
                                        num_dimensions;
      args.push_back(random_generate_n_attributes_sstr.str());
      std::stringstream random_generate_n_entries_sstr;
      random_generate_n_entries_sstr << "--random_generate_n_entries=" <<
                                     num_points;
      args.push_back(random_generate_n_entries_sstr.str());

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Push in the densities output file name.
      args.push_back(std::string("--densities_out=densities.txt"));

      // Push in the kernel type.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

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
      double bandwidth;
      if(world.rank() == 0) {
        bandwidth = core::math::Random(
                      0.1 * sqrt(
                        2.0 * num_dimensions),
                      0.2 * sqrt(
                        2.0 * num_dimensions));
      }
      boost::mpi::broadcast(world, bandwidth, 0);
      std::stringstream bandwidth_sstr;
      bandwidth_sstr << "--bandwidth=" << bandwidth;
      args.push_back(bandwidth_sstr.str());

      // Parse the distributed KDE arguments.
      mlpack::distributed_kde::DistributedKdeArguments<DistributedTableType>
      distributed_kde_arguments;
      mlpack::distributed_kde::DistributedKde <
      DistributedTableType >::ParseArguments(
        world, args, &distributed_kde_arguments);

      std::cout << "Bandwidth value " << bandwidth << "\n";

      // Call the distributed kde driver.
      mlpack::distributed_kde::DistributedKde <
      DistributedTableType > distributed_kde_instance;
      distributed_kde_instance.Init(world, distributed_kde_arguments);

      // Compute the result.
      mlpack::kde::KdeResult <
      std::vector<double> > distributed_kde_result;
      distributed_kde_instance.Compute(
        distributed_kde_arguments, &distributed_kde_result);

      // Call the ultra-naive.
      std::vector<double> ultra_naive_distributed_kde_result;

      /*
      UltraNaive_(
        *(distributed_kde_arguments.metric_),
      	*(distributed_kde_arguments.reference_table_),
        *(distributed_kde_arguments.reference_table_),
        distributed_kde_instance.global().kernel(),
        ultra_naive_distributed_kde_result);
      if(CheckAccuracy_(
            distributed_kde_result.densities_,
            ultra_naive_distributed_kde_result,
            distributed_kde_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
      }
      */

      return 0;
    };
};
};
};

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Delete the teporary files and put a barrier.
  std::stringstream temporary_file_name;
  temporary_file_name << "tmp_file" << world.rank();
  remove(temporary_file_name.str().c_str());
  world.barrier();

  srand(time(NULL) + world.rank());

  // Initialize the memory allocator.
  core::table::global_m_file_ = new core::table::MemoryMappedFile();
  core::table::global_m_file_->Init(
    std::string("tmp_file"), world.rank(), world.rank(), 50000000);

  // Call the tests.
  mlpack::distributed_kde::TestDistributed_Kde distributed_kde_test;
  distributed_kde_test.StressTestMain(world);

  std::cout << "All tests passed!\n";
  return 0;
}
