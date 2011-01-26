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

/** @brief The test driver for the distributed kde.
 */
class TestDistributed_Kde {

  private:

    typedef core::tree::GenMetricTree<mlpack::kde::KdeStatistic> TreeSpecType;

    typedef core::table::DistributedTable < TreeSpecType > DistributedTableType;

    typedef DistributedTableType::TableType TableType;

    void CopyTable_(TableType *local_table, double **start_ptrs) {
      int n_attributes = local_table->n_attributes();

      // Look at old_from_new_indices.
      std::pair<int, std::pair<int, int> > *old_from_new =
        local_table->old_from_new();

      for(int i = 0; i < local_table->n_entries(); i++) {
        core::table::DensePoint point;
        int old_process_index = old_from_new[i].first;
        int old_process_point_index = old_from_new[i].second.first;
        int new_process_old_from_new_index =
          old_from_new[i].second.second;
        local_table->get(new_process_old_from_new_index, &point);
        double *destination = start_ptrs[old_process_index] +
                              n_attributes * old_process_point_index;
        memcpy(destination, point.ptr(), sizeof(double) * n_attributes);
      }
    }

    void CombineTables_(
      boost::mpi::communicator &world,
      DistributedTableType *distributed_reference_table,
      TableType &output_table, int **total_distribution_in) {
      int total_num_points = 0;
      TableType *local_table = distributed_reference_table->local_table();
      int n_attributes = local_table->n_attributes();
      double **start_ptrs = NULL;

      // The master process needs to figure out the layout of the
      // original tables.
      int *point_distribution = new int[world.size()];
      *total_distribution_in = new int[world.size()];
      int *total_distribution = *total_distribution_in;
      for(int i = 0; i < world.size(); i++) {
        point_distribution[i] = total_distribution[i] = 0;
      }
      std::pair<int, std::pair<int, int> > *old_from_new =
        local_table->old_from_new();
      for(int i = 0; i < local_table->n_entries(); i++) {
        point_distribution[ old_from_new[i].first ]++;
      }
      boost::mpi::all_reduce(
        world, point_distribution, world.size(),
        total_distribution, std::plus<int>());

      // The master process initializes the global table and the
      // copies its own data onto it.
      if(world.rank() == 0) {
        start_ptrs = new double *[world.size()];
        for(int i = 0; i < world.size(); i++) {
          total_num_points += total_distribution[i];
        }
        output_table.Init(n_attributes, total_num_points);
        start_ptrs[0] = output_table.data().ptr();
        total_num_points = 0;
        for(int i = 0; i < world.size(); i++) {
          total_num_points += total_distribution[i];
          if(i + 1 < world.size()) {
            start_ptrs[i + 1] =
              start_ptrs[0] + total_num_points * n_attributes;
          }
        }
        CopyTable_(local_table, start_ptrs);
      }

      for(int i = 1; i < world.size(); i++) {
        TableType received_table;
        if(world.rank() == 0) {
          // Receive the table from $i$-th process and copy.
          world.recv(i, boost::mpi::any_tag, received_table);
          CopyTable_(&received_table, start_ptrs);
        }
        else {
          // Send the table to the master process.
          world.send(0, i, *local_table);
          break;
        }
      }
      if(world.rank() == 0) {
        delete[] start_ptrs;
      }
      delete[] point_distribution;
    }

    void FindOriginalIndices_(
      std::pair<int, std::pair<int, int> > *old_from_new,
      int n_entries, int new_point_index,
      int *process_index, int *point_index) {
      for(int i = 0; i < n_entries; i++) {
        if(new_point_index == old_from_new[i].second.second) {
          *process_index = old_from_new[i].first;
          *point_index = old_from_new[i].second.first;
          return;
        }
      }
    }

    bool CheckAccuracy_(
      boost::mpi::communicator &world,
      std::pair<int, std::pair<int, int> > *old_from_new,
      int n_entries,
      int *total_distribution,
      const std::vector<double> &query_results,
      const std::vector<double> &naive_query_results,
      double relative_error) {

      std::vector<int> cumulative_distribution(world.size(), 0);
      for(unsigned int i = 1; i < cumulative_distribution.size(); i++) {
        cumulative_distribution[i] = cumulative_distribution[i - 1] +
                                     total_distribution[i - 1];
      }

      // Compute the collective L1 norm of the products.
      double achieved_error = 0;
      for(unsigned int j = 0; j < query_results.size(); j++) {
        int process_index = 0;
        int point_index = 0;
        FindOriginalIndices_(
          old_from_new, n_entries, j, &process_index, &point_index);
        int naive_index = cumulative_distribution[process_index] + point_index;
        double per_relative_error =
          fabs(naive_query_results[naive_index] - query_results[j]) /
          fabs(naive_query_results[naive_index]);
        achieved_error = std::max(achieved_error, per_relative_error);
        if(relative_error < per_relative_error) {
          std::cout << query_results[j] << " against " <<
                    naive_query_results[naive_index] << ": " <<
                    per_relative_error << "\n";
        }
      }
      std::cout << "Process " << world.rank() <<
                " achieved a relative error of " << achieved_error << "\n";
      return achieved_error <= relative_error;
    }

    template<typename MetricType>
    void UltraNaive_(
      const MetricType &metric_in,
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
      for(int i = 0; i < 10; i++) {
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
      int num_points = core::math::RandInt(3000, 5000);
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
      std::string references_in("random_dataset.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Push in the densities output file name.
      args.push_back(std::string("--densities_out=densities.txt"));

      // Push in the kernel type.
      if(world.rank() == 0) {
        std::cout << "\n==================\n";
        std::cout << "Test trial begin\n";
        std::cout << "Number of dimensions: " << num_dimensions << "\n";
        fflush(stdout);
        fflush(stderr);
      }
      std::cout << "Number of points generated by " <<
                world.rank() << ": " << num_points << "\n";

      switch(kernel_type) {
        case 0:
          if(world.rank() == 0) {
            std::cout << "Epan kernel, \n";
          }
          args.push_back(std::string("--kernel=epan"));
          break;
        case 1:
          if(world.rank() == 0) {
            std::cout << "Gaussian kernel, \n";
          }
          args.push_back(std::string("--kernel=gaussian"));
          break;
      }

      // Push in the leaf size.
      int leaf_size = core::math::RandInt(15, 25);
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

      // Push in the randomly generate work parameters.
      double max_num_levels_to_serialize;
      double max_num_work_to_dequeue_per_stage;
      if(world.rank() == 0) {
        max_num_levels_to_serialize = core::math::RandInt(3, 7);
        max_num_work_to_dequeue_per_stage = core::math::RandInt(3, 10);
      }
      boost::mpi::broadcast(world, max_num_levels_to_serialize, 0);
      boost::mpi::broadcast(world, max_num_work_to_dequeue_per_stage, 0);
      std::stringstream max_num_levels_to_serialize_sstr;
      std::stringstream max_num_work_to_dequeue_per_stage_sstr;
      max_num_levels_to_serialize_sstr
          << "--max_num_levels_to_serialize_in=" << max_num_levels_to_serialize;
      max_num_work_to_dequeue_per_stage_sstr
          << "--max_num_work_to_dequeue_per_stage_in=" <<
          max_num_work_to_dequeue_per_stage;
      args.push_back(max_num_levels_to_serialize_sstr.str());
      args.push_back(max_num_work_to_dequeue_per_stage_sstr.str());

      // Parse the distributed KDE arguments.
      mlpack::distributed_kde::DistributedKdeArguments <
      DistributedTableType > distributed_kde_arguments;
      mlpack::distributed_kde::DistributedKde <
      DistributedTableType >::ParseArguments(
        world, args, &distributed_kde_arguments);

      if(world.rank() == 0) {
        std::cout << "Bandwidth value " << bandwidth << "\n";
      }
      world.barrier();

      // Call the distributed kde driver.
      mlpack::distributed_kde::DistributedKde <
      DistributedTableType > distributed_kde_instance;
      distributed_kde_instance.Init(world, distributed_kde_arguments);

      // Compute the result.
      mlpack::kde::KdeResult <
      std::vector<double> > distributed_kde_result;
      distributed_kde_instance.Compute(
        distributed_kde_arguments, &distributed_kde_result);

      // For each process, check whether all the othe reference points
      // have been encountered.
      DistributedTableType *distributed_reference_table =
        distributed_kde_arguments.reference_table_;
      int total_num_points = -1;
      for(int i = 0; i < world.size(); i++) {
        total_num_points += distributed_reference_table->local_n_entries(i);
      }
      for(unsigned int i = 0; i < distributed_kde_result.pruned_.size(); i++) {
        if(distributed_kde_result.pruned_[i] != total_num_points) {
          std::cerr << "Not all reference point have been accounted for.\n";
          std::cerr << "Got " << distributed_kde_result.pruned_[i] <<
                    " instead of " << total_num_points << "\n";
          exit(-1);
        }
      }

      // Call the ultra-naive.
      std::vector<double> ultra_naive_distributed_kde_result;

      // The master collects all the distributed tables and collects a
      // mega-table for which can be used to compute the naive
      // results.
      TableType combined_reference_table;
      int *total_distribution;
      CombineTables_(
        world, distributed_reference_table, combined_reference_table,
        &total_distribution);

      if(world.rank() == 0) {
        UltraNaive_(
          *(distributed_kde_arguments.metric_),
          combined_reference_table, combined_reference_table,
          distributed_kde_instance.global().kernel(),
          ultra_naive_distributed_kde_result);
      }

      // The master broadcasts the ultranaive result to all processes,
      // each of which checks against it.
      boost::mpi::broadcast(world, ultra_naive_distributed_kde_result, 0);

      if(CheckAccuracy_(
            world,
            distributed_reference_table->local_table()->old_from_new(),
            distributed_reference_table->n_entries(),
            total_distribution,
            distributed_kde_result.densities_,
            ultra_naive_distributed_kde_result,
            distributed_kde_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
        exit(-1);
      }
      world.barrier();

      return 0;
    }
};
}
}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Delete the teporary files and put a barrier.
  std::stringstream temporary_file_name;
  temporary_file_name << "tmp_file" << world.rank();
  remove(temporary_file_name.str().c_str());
  world.barrier();
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());

  // Call the tests.
  mlpack::distributed_kde::TestDistributed_Kde distributed_kde_test;
  distributed_kde_test.StressTestMain(world);

  if(world.rank() == 0) {
    std::cout << "All tests passed!\n";
  }
  fflush(stdout);
  fflush(stderr);
  world.barrier();
  return 0;
}
