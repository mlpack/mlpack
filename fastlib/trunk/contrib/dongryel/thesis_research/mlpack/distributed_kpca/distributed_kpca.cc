/** @file distributed_kpca.cc
 *
 *  The driver for the distributed kernel PCA.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/tree/gen_kdtree.h"
#include "core/math/math_lib.h"
#include "mlpack/distributed_kpca/distributed_kpca_dev.h"
#include "mlpack/distributed_kpca/kpca_result.h"

template<typename KernelType>
void StartComputation(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a kd-tree.
  typedef core::table::DistributedTable <
  core::tree::GenKdTree <
  core::tree::AbstractStatistic > > DistributedTableType;

  // Parse arguments for kernel PCA.
  mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > distributed_kpca_arguments;
  if(mlpack::distributed_kpca::
      DistributedKpcaArgumentParser::ParseArguments(
        world, vm, &distributed_kpca_arguments)) {
    return;
  }

  // Instantiate a distributed kernel PCA object.
  mlpack::distributed_kpca::DistributedKpca <
  DistributedTableType, KernelType > distributed_kernel_pca_instance;
  distributed_kernel_pca_instance.Init(world, distributed_kpca_arguments);

  // Compute the result.
  mlpack::distributed_kpca::KpcaResult kernel_pca_result;
  distributed_kernel_pca_instance.Compute(
    distributed_kpca_arguments, &kernel_pca_result);

  // Output the kernel PCA result to the file.
  kernel_pca_result.Print(distributed_kpca_arguments.kpca_components_out_);
}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Seed the random number.
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  boost::program_options::variables_map vm;
  if(mlpack::distributed_kpca::
      DistributedKpcaArgumentParser::ConstructBoostVariableMap(
        world, argc, argv, &vm)) {
    return 0;
  }

  // Do a quick peek at the kernel and expansion type.
  std::string kernel_type = vm["kernel"].as<std::string>();

  if(kernel_type == "gaussian") {
    StartComputation <
    core::metric_kernels::GaussianKernel > (world, vm);
  }

  return 0;
}
