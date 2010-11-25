/** @file distributed_auction.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_AUCTION_H
#define CORE_TABLE_DISTRIBUTED_AUCTION_H

#include <boost/mpi.hpp>

namespace core {
namespace table {
class DistributedAuction {
  private:

    int ComputeBid_(
      const std::vector<int> &prices, const std::vector<int> &weights,
      double threshold_in, int *best_item_index_out) const {

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
      for(unsigned int i = 2; i < weights.size(); i++) {
        // If larger than the current maximum, then the second maximum
        // changes as well.
        int current_difference = weights[i] - prices[i];
        if(current_difference >= best_difference) {
          second_best_difference = best_difference;
          second_best_item_index = best_item_index;
          best_difference = current_difference;
          best_item_index = i;
        }
        else if(current_difference >= second_best_difference) {
          second_best_difference = current_best_difference;
          second_best_item_index = i;
        }
      }
      int bid = best_difference - second_best_difference + threshold_in;
      *best_item_index_out = best_item_index;

      return bid;
    }

  public:

    int Assign(
      boost::mpi::communicator &comm, const std::vector<int> &weights,
      double threshold_in) {

      // The price of each item.
      std::vector<int> prices(comm.size(), 0);

      // The list of bids (to be used by the master process).
      std::vector< std::pair<int, int> > list_of_bids;

      // The temporary space to be used for resolving the best bids
      // for each item.
      std::vector< std::pair<int, int> > best_bid_per_item;

      // The current assignment.
      int current_assignment = -1;

      // The main loop of the algorithm.
      do {

        int bid = -1;
        int bid_item_index = -1;

        if(current_assignment < 0) {

          // Loop through and find out the appropriate bid.
          bid = ComputeBid_(prices, weights, threshold_in, &bid_item_index);
        }

        // The master gathers all the bids.
        boost::mpi::gather(
          comm, std::pair<int, int>(bid_item_index, bid), list_of_bids, 0);

        // The master decides the assignment.
        if(comm.rank() == 0) {
          best_bid_per_item.resize(comm.size());
          for(unsigned int i = 0; i < best_bid_per_item.size(); i++) {
            best_bid_per_item[i].first = -1;
            best_bid_per_item[i].second = -1;
          }
          for(unsigned int i = 0; i < list_of_bids.size(); i++) {
            bid_item_index = list_of_bids[i].first;
            bid = list_of_bids[i].second;
            if(bid_item_index >= 0 &&
                bid > best_bid_per_item[bid_item_index].second) {
              best_bid_per_item[bid_item_index].first = i;
              best_bid_per_item[bid_item_index].second = bid;
            }
          }

        }

        // Synchronize.
        comm.barrier();

        // Send back the updated prices to each process.
        boost::mpi::broadcast(comm, prices, 0);
      }
      while(true);

      return current_assignment;
    }
};
};
};

#endif
