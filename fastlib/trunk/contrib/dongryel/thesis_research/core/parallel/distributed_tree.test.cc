/** @file distributed_tree.test.cc
 *
 *  A "stress" test driver for distributed trees.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <boost/mpi.hpp>
#include <time.h>
#include "core/metric_kernels/lmetric.h"
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

    int StressTestMain(boost::mpi::communicator &world) {
      for(int i = 0; i < 10; i++) {

        // Only the master broadcasts the dimension;
        int num_dimensions;
        if(world.rank() == 0) {
          num_dimensions = core::math::RandInt(2, 3);
        }
        boost::mpi::broadcast(world, num_dimensions, 0);
        int num_points = core::math::RandInt(3000, 8000);
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
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      random_table.Save(references_in);

      DistributedTableType distributed_table;
      distributed_table.Init(references_in, world);
      core::parallel::VanillaDistributedTreeBuilder <
      DistributedTableType > builder;
      builder.Init(distributed_table);
      core::metric_kernels::LMetric<2> l2_metric;
      int leaf_size = core::math::RandInt(20, 40);
      builder.Build(world, l2_metric, leaf_size);

      return true;
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
  typedef core::table::Table<GenMetricTreeSpecType> GenMetricTreeTableType;
  typedef core::table::DistributedTable<GenMetricTreeSpecType>
  GenMetricTreeDistributedTableType;

  // The general kd tree type.
  typedef core::tree::GenKdTree<core::tree::AbstractStatistic>
  GenKdTreeSpecType;
  typedef core::table::Table<GenKdTreeSpecType> GenKdTreeTableType;
  typedef core::table::DistributedTable<GenKdTreeSpecType>
  GenKdTreeDistributedTableType;

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
