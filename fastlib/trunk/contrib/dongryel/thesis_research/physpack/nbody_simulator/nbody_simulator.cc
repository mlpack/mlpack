#include <iostream>
#include <string>
#include <armadillo>
#include "boost/mpi.hpp"
#include "boost/serialization/string.hpp"
#include "core/math/math_lib.h"
#include "nbody_simulator_dev.h"
#include <vector>
#include <cstdlib>

int main(int argc, char *argv[]) {

  // Parse arguments for Nbody.
  physpack::nbody_simulator::NbodySimulatorArguments nbody_simulator_arguments;
  physpack::nbody_simulator::NbodySimulator::ParseArguments(
    argc, argv, &nbody_simulator_arguments);

  // Instantiate a Nbody object.
  physpack::nbody_simulator::NbodySimulator nbody_simulator_instance;
  nbody_simulator_instance.Init(nbody_simulator_arguments);

  // Compute the result.
  physpack::nbody_simulator::NbodySimulatorResult nbody_simulator_result;
  nbody_simulator_instance.Compute(
    nbody_simulator_arguments, &nbody_simulator_result);

  // Output the Nbody result to the file.
  std::cerr << "Writing the potentials to the file: " <<
            nbody_simulator_arguments.potentials_out_ << "\n";
  nbody_simulator_result.PrintDebug(nbody_simulator_arguments.potentials_out_);
  return 0;
}
