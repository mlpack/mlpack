/** @file distributed_auction.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_AUCTION_H
#define CORE_TABLE_DISTRIBUTED_AUCTION_H

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

namespace core {
namespace table {
class DistributedAuction {
  private:

    class IntDoublePair {
      private:

        friend class boost::serialization::access;

      public:
        int first;

        double second;

      public:

        IntDoublePair() {
          first = -1;
          second = -1;
        }

        IntDoublePair(int first_in, double second_in) {
          first = first_in;
          second = second_in;
        }

        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
          ar & first;
          ar & second;
        }
    };

    double ComputeBid_(
      const std::vector<double> &prices, const std::vector<double> &weights,
      double threshold_in, int *best_item_index_out) const {

      int best_item_index = -1;
      int second_best_item_index = -1;

      // Compute the differences and initialize.
      double best_difference = weights[0] - prices[0];
      double second_best_difference = weights[1] - prices[1];
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
        double current_difference = weights[i] - prices[i];
        if(current_difference >= best_difference) {
          second_best_difference = best_difference;
          second_best_item_index = best_item_index;
          best_difference = current_difference;
          best_item_index = i;
        }
        else if(current_difference >= second_best_difference) {
          second_best_difference = current_difference;
          second_best_item_index = i;
        }
      }
      if(best_difference == second_best_difference) {
        int max_index = std::max(best_item_index, second_best_item_index);
        int min_index = std::min(best_item_index, second_best_item_index);
        best_item_index = min_index;
        second_best_item_index = max_index;
      }
      double bid = best_difference - second_best_difference + threshold_in;
      *best_item_index_out = best_item_index;

      return bid;
    }

  public:

    int Assign(
      boost::mpi::communicator &comm, const std::vector<double> &weights,
      double threshold_in) {

      // The price of each item.
      std::vector<double> prices(comm.size(), 0);

      // The list of bids (to be used by the master process).
      std::vector< IntDoublePair > list_of_bids;

      // The temporary space to be used for resolving the best bids
      // for each item. It maps each item to the id of the process.
      std::vector< IntDoublePair > best_bid_per_item;

      // The first maps the $i$-th process to the id of the item. The
      // second maps the $i$-th item to the id of the process.
      std::vector< std::pair< int, int > > global_assignments(
        weights.size(), std::pair<int, int>(-1, -1));

      // The main loop of the algorithm.
      do {

        double bid = -1;
        int bid_item_index = -1;

        // If the current process has not been able to grab an item,
        if(global_assignments[comm.rank()].first < 0) {

          // Loop through and find out the appropriate bid.
          bid = ComputeBid_(prices, weights, threshold_in, &bid_item_index);
        }

        // The master gathers all the bids.
        boost::mpi::gather(
          comm, IntDoublePair(
            bid_item_index, bid), list_of_bids, 0);

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

          // Now assign.
          for(unsigned int i = 0; i < best_bid_per_item.size(); i++) {
            if(best_bid_per_item[i].first >= 0) {

              // Check if this item has been assigned to another
              // process already, if so un-assign.
              if(global_assignments[i].second >= 0) {
                global_assignments[ global_assignments[i].second ].first = -1;
                global_assignments[i].second = -1;
              }

              // Potentially re-assign.
              global_assignments[ best_bid_per_item[i].first ].first = i;
              global_assignments[i].second = best_bid_per_item[i].first;

              // Update the price.
              prices[i] += best_bid_per_item[i].second;
            }
          }
        }

        // Synchronize.
        comm.barrier();

        // Send back the updated global assignments to each process.
        boost::mpi::broadcast(comm, global_assignments, 0);

        // Send back the updated prices to each process.
        boost::mpi::broadcast(comm, prices, 0);

        // Check whether every item has been assigned.
        bool all_assigned = true;
        for(unsigned int i = 0; all_assigned && i < global_assignments.size();
            i++) {
          all_assigned = all_assigned && (global_assignments[i].first >= 0);
        }
        comm.barrier();

        if(all_assigned) {
          break;
        }
      }
      while(true);

      return global_assignments[ comm.rank()].first;
    }
};
};
};

#endif
