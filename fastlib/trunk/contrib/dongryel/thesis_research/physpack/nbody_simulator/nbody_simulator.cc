/** @file nbody_simulator.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <iostream>
#include <string>
#include <armadillo>
#include "boost/mpi.hpp"
#include "boost/serialization/string.hpp"
#include "core/math/math_lib.h"
#include "core/util/timer.h"
#include "nbody_simulator_dev.h"
#include "core/tree/gen_kdtree.h"
#include <vector>
#include <cstdlib>

int main(int argc, char *argv[]) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table < core::tree::GenKdTree > TableType;

  // Parse arguments for Nbody.
  physpack::nbody_simulator::NbodySimulatorArguments<TableType> nbody_simulator_arguments;
  physpack::nbody_simulator::NbodySimulator<TableType>::ParseArguments(
    argc, argv, &nbody_simulator_arguments);

  if(nbody_simulator_arguments.table_->n_attributes() != 3) {
    std::cerr << "The dimensionality of the points is not three!\n";
    return -1;
  }

  // Instantiate a Nbody object.
  core::util::Timer tree_build_timer;
  tree_build_timer.Start();
  physpack::nbody_simulator::NbodySimulator<TableType> nbody_simulator_instance;
  nbody_simulator_instance.Init(nbody_simulator_arguments);
  tree_build_timer.End();
  std::cout << tree_build_timer.GetTotalElapsedTime() << " seconds spent on "
            "building the tree.\n";

  // Compute the result.
  core::util::Timer compute_timer;
  compute_timer.Start();
  physpack::nbody_simulator::NbodySimulatorResult nbody_simulator_result;
  nbody_simulator_instance.Compute(
    nbody_simulator_arguments, &nbody_simulator_result);
  compute_timer.End();
  std::cout << compute_timer.GetTotalElapsedTime() << " seconds spent on "
            "the potential computation.\n";

  // Print out the prune statistics.
  std::cout << "Deterministic prunes: " <<
            nbody_simulator_result.num_deterministic_prunes_ << "\n";
  std::cout << "Monte Carlo prunes: " <<
            nbody_simulator_result.num_monte_carlo_prunes_ << "\n";

  // Output the Nbody result to the file.
  std::cerr << "Writing the potentials to the file: " <<
            nbody_simulator_arguments.potentials_out_ << "\n";
  nbody_simulator_result.PrintDebug(nbody_simulator_arguments.potentials_out_);
  return 0;
}
