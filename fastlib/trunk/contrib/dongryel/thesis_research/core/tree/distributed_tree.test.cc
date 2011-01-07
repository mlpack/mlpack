/** @file distributed_tree.test.cc
 *
 *  A "stress" test driver for distributed trees.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "boost/test/unit_test.hpp"
#include "core/metric_kernels/lmetric.h"
#include "core/table/distributed_table.h"
#include "core/tree/distributed_tree_builder.h"
#include "core/math/math_lib.h"
#include <time.h>

namespace core {
namespace tree {

template<typename TableType>
class TestDistributedTree {
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
        int num_dimensions = core::math::RandInt(3, 20);
        int num_points = core::math::RandInt(3000, 5001);
        if(StressTest(num_dimensions, num_points) == false) {
          printf("Failed!\n");
          exit(0);
        }
      }
      return 0;
    }

    bool StressTest(int num_dimensions, int num_points) {

      std::vector< std::string > args;

      std::cout << "Number of dimensions: " << num_dimensions << "\n";
      std::cout << "Number of points: " << num_points << "\n";

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Generate the random dataset and save it.
      TableType random_table;
      GenerateRandomDataset_(
        num_dimensions, num_points, &random_table);
      random_table.Save(references_in);

      // Reload the table twice and build the tree on one of them.
      TableType reordered_table;
      reordered_table.Init(references_in);
      TableType original_table;
      original_table.Init(references_in);
      core::metric_kernels::LMetric<2> l2_metric;
      reordered_table.IndexData(l2_metric, 20);
      for(int i = 0; i < reordered_table.n_entries(); i++) {
        core::table::DensePoint reordered_point;
        core::table::DensePoint original_point;
        reordered_table.get(i, &reordered_point);
        original_table.get(i, &original_point);
        for(int j = 0; j < reordered_table.n_attributes(); j++) {
          if(reordered_point[j] != original_point[j]) {
            return false;
          }
        }
      }
      return true;
    }
};
};
};

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Tree type: hard-coded for a metric tree.
  typedef core::tree::GenMetricTree<core::tree::AbstractStatistic> TreeSpecType;
  typedef core::table::Table <TreeSpecType> TableType;
  typedef core::table::DistributedTable<TreeSpecType> DistributedTableType;

  DistributedTableType distributed_table;
  core::tree::DistributedTreeBuilder<DistributedTableType> builder;
  builder.Init(distributed_table, 0.2);
  core::metric_kernels::LMetric<2> l2_metric;
  builder.Build(l2_metric, world);

  // Call the tests.
  core::tree::TestDistributedTree<TableType> tree_test;
  tree_test.StressTestMain();

  std::cout << "All tests passed!\n";
  return 0;
}
