/** @file distributed_tree.test.cc
 *
 *  A "stress" test driver for distributed trees.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <boost/mpi.hpp>
#include <omp.h>
#include <time.h>
#include "core/metric_kernels/lmetric.h"
#include "core/parallel/random_dataset_generator.h"
#include "core/table/empty_query_result.h"
#include "core/table/distributed_table.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/parallel/vanilla_distributed_tree_builder.h"
#include "core/math/math_lib.h"

namespace core {
namespace tree {

template<typename DistributedTableType>
class TestDistributedTree {
  public:
    typedef typename DistributedTableType::TableType TableType;

    typedef typename TableType::TreeType TreeType;

  private:

    bool TestLocalTree_(TreeType *node) {
      if((! node->is_leaf()) && node->count() == 0) {
        printf("A non-leaf node has zero points!\n");
        return false;
      }

      if(! node->is_leaf()) {
        return TestLocalTree_(node->left()) && TestLocalTree_(node->right());
      }
      return true;
    }

  public:

    int StressTestMain(boost::mpi::communicator &world) {
      for(int i = 0; i < 1; i++) {

        // Only the master broadcasts the dimension;
        int num_dimensions;
        if(world.rank() == 0) {
          num_dimensions = core::math::RandInt(10, 11);
        }
        boost::mpi::broadcast(world, num_dimensions, 0);
        int num_points = core::math::RandInt(1000000, 1000001);
        if(StressTest(world, num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
        printf("\n");
      }
      return 0;
    }

    bool StressTest(
      boost::mpi::communicator &world, int num_dimensions, int num_points) {

      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

      // Push in the reference dataset name.
      std::stringstream reference_file_name_sstr;
      reference_file_name_sstr << "random_dataset.csv" << world.rank();
      std::string references_in = reference_file_name_sstr.str();

      // Generate the random dataset and save it.
      TableType random_table;
      core::parallel::RandomDatasetGenerator::Generate(
        num_dimensions, num_points, world.rank(),
        std::string("none"), false, &random_table);
      random_table.Save(references_in);
      int leaf_size = core::math::RandInt(20, 40);
      core::metric_kernels::LMetric<2> l2_metric;

      {
        // First, build with only one thread.
        printf("First building with only one thread...\n");
        omp_set_num_threads(1);
        DistributedTableType distributed_table;
        distributed_table.Init(references_in, world);
        core::parallel::VanillaDistributedTreeBuilder <
        DistributedTableType > builder;
        builder.Init(distributed_table);
        builder.Build(world, l2_metric, leaf_size, 0);
      }

      // Next, build with four threads.
      printf("Now building with four threads...\n");
      omp_set_num_threads(4);
      DistributedTableType distributed_table;
      distributed_table.Init(references_in, world);
      core::parallel::VanillaDistributedTreeBuilder <
      DistributedTableType > builder;
      builder.Init(distributed_table);
      builder.Build(world, l2_metric, leaf_size, 0);

      // Tes the local tree.
      return TestLocalTree_(distributed_table.local_table()->get_tree());
    }
};
}
}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // The general metric tree type.
  typedef core::tree::GenMetricTree<core::tree::AbstractStatistic>
  GenMetricTreeSpecType;
  typedef core::table::DistributedTable <
  GenMetricTreeSpecType, core::table::EmptyQueryResult >
  GenMetricTreeDistributedTableType;
  typedef typename GenMetricTreeDistributedTableType::TableType GenMetricTreeTableType;

  // The general kd tree type.
  typedef core::tree::GenKdTree<core::tree::AbstractStatistic>
  GenKdTreeSpecType;
  typedef core::table::DistributedTable <
  GenKdTreeSpecType, core::table::EmptyQueryResult >
  GenKdTreeDistributedTableType;
  typedef typename GenKdTreeDistributedTableType::TableType GenKdTreeTableType;

  // Call the tests.
  printf("Testing the general metric trees:\n");
  core::tree::TestDistributedTree<GenMetricTreeDistributedTableType>
  gen_metric_tree_test;
  gen_metric_tree_test.StressTestMain(world);
  printf("Testing the general kd trees:\n");
  core::tree::TestDistributedTree<GenKdTreeDistributedTableType>
  gen_kdtree_test;
  gen_kdtree_test.StressTestMain(world);

  std::cout << "All tests passed!\n";
  return 0;
}
