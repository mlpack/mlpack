/** @file distributed_auction.test.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/table/distributed_auction.h"
#include "core/math/math_lib.h"

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  core::table::DistributedAuction auction;

  // Seed the random number generator.
  srand(time(NULL) + world.rank());

  // Create a random vector of weights.
  std::vector<double> weights(world.size(), 0);
  for(unsigned int i = 0; i < weights.size(); i++) {
    weights[i] = core::math::Random(0.5, 5.0);
  }

  // Get the assignments.
  auction.Assign(world, weights, std::numeric_limits<double>::epsilon());

  return 0;
}
