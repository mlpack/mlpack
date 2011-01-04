#include <iostream>
#include <string>
#include <armadillo>
#include "boost/mpi.hpp"
#include "boost/serialization/string.hpp"
#include "core/math/math_lib.h"
#include <vector>
#include <cstdlib>

int main(int argc, char *argv[]) {
  const int num_elements_per_machine = 3;
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());
  if(world.size() <= 1) {
    std::cout << "Please specify a process number greater than 1.\n";
    exit(0);
  }
  if(world.rank() == 0) {
    printf("Number of processes: %d\n", world.size());
  }

  // Generate a set of random numbers for this processor.
  std::vector<int> numbers(num_elements_per_machine, 0);
  printf("Process %d generated: ", world.rank());
  for(int i = 0; i < num_elements_per_machine; i++) {
    numbers[i] = core::math::RandInt(3, 10);
    printf("%d ", numbers[i]);
  }
  printf("\n");

  // The total for this processor.
  int total = numbers[0];
  for(int i = 1; i < num_elements_per_machine; i++) {
    total += numbers[i];
  }
  printf("Process %d got %d\n", world.rank(), total);

  // The collect phase.
  int k;
  for(k = 1; k < world.size(); k *= 2) {
    if((world.rank() & k) == 0) {
      printf("Sending from Process %d to Process %d\n", world.rank(),
             world.rank() + k);
      world.send(world.rank() + k, world.rank(), total);
      break;
    }
    else {
      double received;
      printf("Process %d needs to receive from Process %d\n",
             world.rank(), world.rank() - k);
      boost::mpi::request receive_request =
        world.irecv(world.rank() - k, boost::mpi::any_tag, received);
      receive_request.wait();
      total += received;
    }
  }
  printf("Done with the collect phase...\n");

  // The distribute phase.
  if(world.rank() == world.size() - 1) {
    // Reset the last processor's subtotal to zero.
    total = 0;
  }
  if(k >= world.size()) {
    k /= 2;
  }
  while(k > 0) {
    if((world.rank() & k) == 0) {
      world.send(world.rank() + k, 0, total);
      boost::mpi::request receive_request =
        world.irecv(world.rank() + k, boost::mpi::any_tag, total);
      receive_request.wait();
    }
    else {
      int t;
      boost::mpi::request receive_request =
        world.irecv(world.rank() - k, boost::mpi::any_tag, t);
      world.send(world.rank() - k, 0, total);
      receive_request.wait();
      total = t + total;
    }
    k /= 2;
  }
  printf("Done with the distribute phase...n");

  printf("Processor %d got the preceeding total %d.\n", world.rank(), total);

  // Update array to have the prefix sums.
  for(int i = 0; i < num_elements_per_machine; i++) {
    total += numbers[i];
    numbers[i] = total;
  }

  // For each process, print out its portion.
  printf("Process #%d has: ", world.rank());
  for(int i = 0; i < num_elements_per_machine; i++) {
    printf("%d ", numbers[i]);
  }
  printf("\n");

  return 0;
}
