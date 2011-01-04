/** @file distributed_auction.test.cc
 *
 *  A simple test for distributed auction algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/parallel/distributed_auction.h"
#include "core/math/math_lib.h"

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  core::parallel::DistributedAuction auction;

  // Seed the random number generator.
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());

  // Create a random vector of weights.
  std::vector<double> weights(world.size(), 0);
  for(unsigned int i = 0; i < weights.size(); i++) {
    weights[i] = core::math::Random(0.5, 5.0);
    printf("Weight %d: %g\n", world.rank(), weights[i]);
  }

  // Get the assignments.
  int item_index =
    auction.Assign(world, weights, std::numeric_limits<double>::epsilon());
  printf("Process %d got item %d\n", world.rank(), item_index);

  return 0;
}
