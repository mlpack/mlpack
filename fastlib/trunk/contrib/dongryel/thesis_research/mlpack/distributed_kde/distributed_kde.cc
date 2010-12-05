/** @file distributed_kde.test.cc
 *
 *  The test driver for the distributed kde.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/table/mailbox.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/kde/kde_dualtree.h"
#include "mlpack/distributed_kde/distributed_kde_dev.h"

typedef core::tree::GenMetricTree<mlpack::kde::KdeStatistic> TreeSpecType;
typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;
typedef core::table::Table<TreeSpecType> TableType;
typedef core::table::DistributedTable<TreeSpecType> DistributedTableType;

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // For the moment, assume that the number of processes is a power of
  // two.
  if((world.size() & (world.size() - 1)) != 0) {
    printf(
      "Currently only supports a number of process equal to a power of 2.\n");
    return -1;
  }

  // Delete the teporary files and put a barrier.
  std::stringstream temporary_file_name;
  temporary_file_name << "tmp_file" << world.rank();
  remove(temporary_file_name.str().c_str());
  world.barrier();

  // Initialize the memory allocator.
  core::table::global_m_file_ = new core::table::MemoryMappedFile();
  core::table::global_m_file_->Init(
    std::string("tmp_file"), world.rank(), world.rank(), 5000000);

  // Seed the random number.
  srand(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  // Parse arguments for the distributed kde.
  mlpack::distributed_kde::DistributedKdeArguments<DistributedTableType>
  distributed_kde_arguments;
  mlpack::distributed_kde::DistributedKde<DistributedTableType>::ParseArguments(
    argc, argv, world, &distributed_kde_arguments);

  // Instantiate a distributed KDE object.
  mlpack::distributed_kde::DistributedKde<DistributedTableType>
  distributed_kde_instance;
  distributed_kde_instance.Init(world, distributed_kde_arguments);

  // Compute the result.
  mlpack::kde::KdeResult< std::vector<double> > kde_result;
  distributed_kde_instance.Compute(distributed_kde_arguments, &kde_result);

  // Output the KDE result to the file.
  std::cerr << "Writing the densities to the file: " <<
            distributed_kde_arguments.densities_out_ << "\n";
  // kde_result.PrintDebug(distributed_kde_arguments.densities_out_);

  return 0;
}
