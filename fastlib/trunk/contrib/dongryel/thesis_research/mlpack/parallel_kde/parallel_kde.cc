#include <iostream>
#include <string>
#include <armadillo>
#include "boost/mpi.hpp"
#include "boost/serialization/string.hpp"
#include "core/math/math_lib.h"
#include <vector>
#include <cstdlib>

class Particle {
  private:
    std::vector<int> position_;

    friend class boost::serialization::access;

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & position_;
    }

    const std::vector<int> &position() const {
      return position_;
    }

    Particle() {
      position_.resize(3);
      for(int i = 0; i < position_.size(); i++) {
        position_[i] = 0;
      }
    }

    void AddParticle(const Particle &particle_in) {
      for(unsigned int i = 0; i < 3; i++) {
        position_[i] += particle_in.position()[i];
      }
    }

    void Randomize() {
      for(unsigned int i = 0; i < 3; i++) {
        position_[i] = core::math::RandInt(0, 3);
      }
    }

    void Print() const {
      std::cout << position_[0] << " " << position_[1] <<
                " " << position_[2] << "\n";
    }
};

int main(int argc, char *argv[]) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  if(world.size() <= 1) {
    std::cout << "Please specify a process number greater than 1.\n";
    exit(0);
  }

  Particle generated_particle;

  // Broadcast the skeleton from the master to the slaves.
  broadcast(world, boost::mpi::skeleton(generated_particle), 0);

  boost::mpi::content c = boost::mpi::get_content(generated_particle);

  // The master process.
  if(world.rank() == 0) {
    std::vector<int> num_messages(world.size(), 0);
    for(int i = 1; i < world.size(); i++) {
      num_messages[i] = core::math::RandInt(3, 8);

      // Send the number of messages to receive per each processor.
      world.isend(i, 0, num_messages[i]);
      printf("Processor 0 is sending processor %d %d message.\n", i,
             num_messages[i]);
    }
    for(int i = 1; i < world.size(); i++) {
      for(int j = 0; j < num_messages[i]; j++) {
        generated_particle.Randomize();
        std::cout << "Master generated and sending to " << i << ": ";
        generated_particle.Print();
        world.isend(i, 1, c);
      }
    }
  }

  // The slave process.
  else {
    int num_messages_to_receive;
    boost::mpi::request main_request =
      world.irecv(0, 0, num_messages_to_receive);
    main_request.wait();

    Particle received_particle;
    for(int j = 0; j < num_messages_to_receive; j++) {
      // Receive from the master.
      boost::mpi::request receive_request = world.irecv(0, 1, c);
      receive_request.wait();
      std::cout << "Process " << world.rank() << " received: ";
      generated_particle.Print();
      received_particle.AddParticle(generated_particle);
    }
    std::cout << "Process " << world.rank() << " got: ";
    received_particle.Print();
  }

  return 0;
}
