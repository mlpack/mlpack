/** @file distributed_local_regression.cc
 *
 *  The driver for the distributed local regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include "mlpack/local_regression/local_regression_dualtree.h"
#include "mlpack/distributed_local_regression/distributed_local_regression_dev.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename KernelAuxType>
void StartComputation(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::DistributedTable <
  core::tree::GenKdTree <
  mlpack::local_regression::LocalRegressionStatistic <
  KernelAuxType::ExpansionType > > > DistributedTableType;

  // Parse arguments for local regression.
  mlpack::distributed_local_regression::DistributedLocalRegressionArguments <
  DistributedTableType > distributed_local_regression_arguments;
  if(mlpack::distributed_local_regression::
      DistributedLocalRegressionArgumentParser::ParseArguments(
        world, vm, &distributed_local_regression_arguments)) {
    return;
  }

  // Instantiate a distributed local regression object.
  mlpack::distributed_local_regression::DistributedLocalRegression <
  DistributedTableType, KernelAuxType > distributed_local_regression_instance;
  distributed_local_regression_instance.Init(world, distributed_local_regression_arguments);

  // Compute the result.
  mlpack::local_regression::LocalRegressionResult local_regression_result;
  distributed_local_regression_instance.Compute(
    distributed_local_regression_arguments, &local_regression_result);

  // Output the local regression result to the file.
  std::cerr << "Writing the densities to the file: " <<
            distributed_local_regression_arguments.densities_out_ << "\n";
  local_regression_result.Print(
    distributed_local_regression_arguments.densities_out_);
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
  if(mlpack::distributed_local_regression::
      DistributedLocalRegressionArgumentParser::ConstructBoostVariableMap(
        world, argc, argv, &vm)) {
    return 0;
  }

  // Do a quick peek at the kernel and expansion type.
  std::string kernel_type = vm["kernel"].as<std::string>();
  std::string series_expansion_type =
    vm["series_expansion_type"].as<std::string>();

  if(kernel_type == "gaussian") {
    if(series_expansion_type == "hypercube") {
      StartComputation <
      mlpack::series_expansion::GaussianKernelHypercubeAux > (world, vm);
    }
    else {
      StartComputation <
      mlpack::series_expansion::GaussianKernelMultivariateAux > (world, vm);
    }
  }
  else {

    // Only the multivariate expansion is available for the
    // Epanechnikov.
    StartComputation <
    mlpack::series_expansion::EpanKernelMultivariateAux > (world, vm);
  }
  return 0;
}
