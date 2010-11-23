/** @file distributed_hungarian.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_HUNGARIAN_H
#define CORE_TABLE_DISTRIBUTED_HUNGARIAN_H

#include <boost/mpi.hpp>

namespace core {
namespace table {
class DistributedHungarian {
  private:

    int ComputeBid_(
      const std::vector<int> &prices, const std::vector<int> &weights) const {

      int best_item_index = -1;
      int second_best_item_index = -1;

      // Compute the differences and initialize.
      int best_difference = weights[0] - prices[0];
      int second_best_difference = weights[1] - prices[1];
      if(best_difference > second_best_difference) {
        best_item_index = 0;
        second_best_item_index = 1;
      }
      else {
        std::swap(best_difference, second_best_difference);
        best_item_index = 1;
        second_best_item_index = 0;
      }


      return bid;
    }

  public:

    int Assign(
      boost::mpi::communicator &comm, const std::vector<int> &weights) {

      // The price of each item.
      std::vector<int> prices(comm.size(), 0);

      // The current assignment.
      int current_assignment = -1;

      // The main loop of the algorithm.
      do {

        if(current_assignment < 0) {

          // Loop through and find out the appropriate bid.
          int bid = ComputeBid_(prices, weights);
        }

        // Wait until every MPI process gets here.
        comm.barrier();

        // The master decides the assignment.
        if(comm.rank() == 0) {

        }

      }
      while(true);

      return current_assignment;
    }
};
};
};

#endif
