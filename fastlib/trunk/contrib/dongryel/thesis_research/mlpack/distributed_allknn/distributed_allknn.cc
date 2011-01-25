/** @file distributed_allknn.cc
 *
 *  The driver for the distributed allknn.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include "mlpack/allknn/allknn_dualtree.h"
#include "mlpack/distributed_allknn/distributed_allknn_dev.h"

typedef core::tree::GenMetricTree<mlpack::allknn::AllknnStatistic> TreeSpecType;
typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;
typedef core::table::Table<TreeSpecType> TableType;
typedef core::table::DistributedTable<TreeSpecType> DistributedTableType;

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Seed the random number.
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  // Parse arguments for the distributed allknn.
  mlpack::distributed_allknn::DistributedAllknnArguments<DistributedTableType>
  distributed_allknn_arguments;
  mlpack::distributed_allknn::DistributedAllknn <
  DistributedTableType >::ParseArguments(
    argc, argv, world, &distributed_allknn_arguments);

  // Instantiate a distributed allknn object.
  mlpack::distributed_allknn::DistributedAllknn<DistributedTableType>
  distributed_allknn_instance;
  distributed_allknn_instance.Init(world, distributed_allknn_arguments);

  // Compute the result.
  mlpack::allknn::AllknnResult< std::vector<double> > allknn_result;
  distributed_allknn_instance.Compute(
    distributed_allknn_arguments, &allknn_result);

  // Output the allknn result to the file.
  std::cerr << "Writing the nearest neighbor indices and distances " <<
            "to the file: " <<
            distributed_allknn_arguments.densities_out_ << "\n";
  allknn_result.PrintDebug(distributed_allknn_arguments.densities_out_);

  return 0;
}
