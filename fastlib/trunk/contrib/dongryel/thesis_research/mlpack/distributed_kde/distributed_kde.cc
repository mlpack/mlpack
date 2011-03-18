/** @file distributed_kde.cc
 *
 *  The driver for the distributed kde.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include "mlpack/kde/kde_dualtree.h"
#include "mlpack/distributed_kde/distributed_kde_dev.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename KernelAuxType>
void StartComputation(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::DistributedTable <
  core::tree::GenMetricTree <
  mlpack::kde::KdeStatistic <
  KernelAuxType::ExpansionType > > > DistributedTableType;

  // Parse arguments for Kde.
  mlpack::distributed_kde::DistributedKdeArguments <
  DistributedTableType > distributed_kde_arguments;
  if(mlpack::distributed_kde::
      DistributedKdeArgumentParser::ParseArguments(
        world, vm, &distributed_kde_arguments)) {
    return;
  }

  // Instantiate a distributed KDE object.
  mlpack::distributed_kde::DistributedKde <
  DistributedTableType, KernelAuxType > distributed_kde_instance;
  distributed_kde_instance.Init(world, distributed_kde_arguments);

  // Compute the result.
  mlpack::kde::KdeResult< std::vector<double> > kde_result;
  distributed_kde_instance.Compute(
    distributed_kde_arguments, &kde_result);

  // Output the KDE result to the file.
  std::cerr << "Writing the densities to the file: " <<
            distributed_kde_arguments.densities_out_ << "\n";
  kde_result.Print(distributed_kde_arguments.densities_out_);
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
  if(mlpack::distributed_kde::
      DistributedKdeArgumentParser::ConstructBoostVariableMap(
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
