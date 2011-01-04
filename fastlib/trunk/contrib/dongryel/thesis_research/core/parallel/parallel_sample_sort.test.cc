/** @file parallel_sample_sort.test.cc
 *
 *  A simple test for distributed parallel sample sort algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/parallel/parallel_sample_sort.h"
#include "core/math/math_lib.h"

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Seed the random number generator.
  srand(time(NULL) + world.rank());

  // Create a random vector of weights.
  std::vector<double> weights(world.size(), 0);
  for(unsigned int i = 0; i < weights.size(); i++) {
    weights[i] = core::math::Random(0.5, 5.0);
    printf("Weight %d: %g\n", world.rank(), weights[i]);
  }

  // Sort.
  core::parallel::ParallelSampleSort<double> sorter;
  sorter.Init(weights, 0.2);

  return 0;
}
