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
#include "core/parallel/distributed_tree_builder.h"
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
      for(int i = 0; i < 1; i++) {

        // Only the master broadcasts the dimension;
        int num_dimensions;
        if(world.rank() == 0) {
          num_dimensions = core::math::RandInt(3, 8);
        }
        boost::mpi::broadcast(world, num_dimensions, 0);
        int num_points = core::math::RandInt(3, 8);
        if(StressTest(world, num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
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
      core::parallel::DistributedTreeBuilder<DistributedTableType> builder;
      builder.Init(distributed_table, 0.2);
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

  // Tree type: hard-coded for a metric tree.
  typedef core::tree::GenMetricTree<core::tree::AbstractStatistic> TreeSpecType;
  typedef core::table::Table <TreeSpecType> TableType;
  typedef core::table::DistributedTable<TreeSpecType> DistributedTableType;

  // Call the tests.
  core::tree::TestDistributedTree<DistributedTableType> tree_test;
  tree_test.StressTestMain(world);


  arma::vec first_point;
  arma::vec second_point;
  first_point.zeros(5);
  second_point.zeros(5);
  for(int i = 0; i < 5; i++) {
    first_point[i] = 0.1;
    second_point[i] = 0.7;
  }
  first_point.print();
  second_point.print();
  printf("Hey: %d\n", core::math::MortonOrderPoints(first_point, second_point));

  std::cout << "All tests passed!\n";
  return 0;
}
