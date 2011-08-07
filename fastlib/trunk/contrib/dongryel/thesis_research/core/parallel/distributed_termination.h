/** @file distributed_termination.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_TERMINATION_H
#define CORE_PARALLEL_DISTRIBUTED_TERMINATION_H

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <deque>
#include "core/parallel/message_tag.h"
#include "core/parallel/route_request.h"

namespace core {
namespace parallel {
class DistributedTermination {

  public:
    typedef core::parallel::RouteRequest<unsigned long int> RouteRequestType;

  private:

    unsigned long int remaining_computation_;

    unsigned long int local_work_;

    std::vector< RouteRequestType > queued_up_completed_computation_;

    /** @brief The number of cache blocks per process.
     */
    int num_cache_blocks_;

    std::vector <
    std::pair <
    RouteRequestType, boost::mpi::request > > messages_to_receive_;

    std::vector< int > receive_cache_locks_;

    std::vector<int> free_slots_for_receiving_;

    std::vector <
    boost::tuple <
    RouteRequestType, boost::mpi::request > > messages_to_send_;

    std::vector<int> sending_in_progress_;

    std::vector<int> free_slots_for_sending_;

    std::vector<int> queue_up_for_sending_from_receive_slots_;

  public:

    void LockCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        receive_cache_locks_[ cache_id ] += num_times;
      }
    }

    void ReleaseCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        receive_cache_locks_[ cache_id ] -= num_times;

        // If the slot is not needed, free it.
        if(receive_cache_locks_[ cache_id ] == 0) {
          free_slots_for_receiving_.push_back(cache_id);
        }
      }
    }

    void push_completed_computation(
      boost::mpi::communicator &comm, unsigned long int quantity_in) {

      // Subtract from the self and queue up a route message so that
      // it can be passed to all the other processes.
      remaining_computation_ -= quantity_in;
      local_work_ -= quantity_in;
      if(comm.size() > 1) {
        if(queued_up_completed_computation_.size() == 0) {
          RouteRequestType new_route_request;
          new_route_request.Init(comm);
          new_route_request.object() = quantity_in;
          new_route_request.add_destinations(comm);
          queued_up_completed_computation_.push_back(new_route_request);
        }
        else {
          queued_up_completed_computation_.back().object() += quantity_in;
        }
      }
    }

    bool can_terminate(boost::mpi::communicator &world) const {

      // Safely terminate when the current process has confirmed that
      // all computation has been completed.
      return remaining_computation_ == 0 &&
             queued_up_completed_computation_.size() == 0 &&
             free_slots_for_sending_.size() == messages_to_send_.size() &&
             free_slots_for_receiving_.size() == messages_to_receive_.size();
    }

    template<typename DistributedTableType>
    void Init(
      boost::mpi::communicator &world,
      DistributedTableType *query_table_in,
      DistributedTableType *reference_table_in,
      int num_cache_blocks_in) {

      unsigned long int total_num_query_points = 0;
      unsigned long int total_num_reference_points = 0;
      for(int i = 0; i < world.size(); i++) {
        total_num_query_points += query_table_in->local_n_entries(i);
        total_num_reference_points += reference_table_in->local_n_entries(i);
      }

      // Initialize the remaining computation.
      remaining_computation_ =
        static_cast<unsigned long int>(total_num_query_points) *
        static_cast<unsigned long int>(total_num_reference_points);
      local_work_ =
        query_table_in->local_n_entries(world.rank()) *
        total_num_reference_points;

      // Compute the number of cache blocks allocated per process.
      num_cache_blocks_ = num_cache_blocks_in;

      // Preallocate the send and receive caches.
      messages_to_receive_.resize(num_cache_blocks_ * world.size());
      messages_to_send_.resize(num_cache_blocks_);
      for(unsigned int i = 0; i < messages_to_send_.size(); i++) {
        free_slots_for_sending_.push_back(i);
      }
      for(unsigned int i = 0; i < messages_to_receive_.size(); i++) {
        free_slots_for_receiving_.push_back(i);
      }

      // Initialize the locks.
      receive_cache_locks_.resize(free_slots_for_receiving_.size());
      std::fill(
        receive_cache_locks_.begin(), receive_cache_locks_.end(), 0);
    }

    void AsynchForwardTerminationMessages(boost::mpi::communicator &world) {

      if(world.size() == 1) {
        return;
      }

      // Queue up send requests by transfering from the receive route
      // requests.
      for(int i = 0; i < static_cast<int>(
            queue_up_for_sending_from_receive_slots_.size()) ; i++) {

        // Examine the back of the route request list.
        int cache_id = queue_up_for_sending_from_receive_slots_[i];
        std::pair <
        RouteRequestType, boost::mpi::request > &route_request_pair =
          messages_to_receive_[ cache_id ];
        RouteRequestType &route_request = route_request_pair.first;

        if(route_request_pair.second.test()) {

          // Release from the cache number of times equal to the
          // number of messages routed.
          this->ReleaseCache(
            cache_id, route_request_pair.first.num_routed());

          // Remove from the queue up list, if done.
          if(route_request.num_destinations() == 0) {
            queue_up_for_sending_from_receive_slots_[i] =
              queue_up_for_sending_from_receive_slots_.back();
            queue_up_for_sending_from_receive_slots_.pop_back();
            i--;
          }

          // Otherwise, queue up another send.
          else {
            route_request_pair.second =
              world.isend(
                route_request.next_destination(world),
                core::parallel::MessageTag::FINISHED_TUPLES,
                route_request) ;
          }
        }
      }

      // Queue up send requests by dequeuing from the list of
      // reference subtrees to send from the current process.
      while(free_slots_for_sending_.size() > 0 &&
            queued_up_completed_computation_.size() > 0) {

        // Examine the back of the route request list.
        RouteRequestType &route_request =
          queued_up_completed_computation_.back();

        // If the next destination is valid, then get a free slot and
        // issue an asynchronous send.
        if(route_request.num_destinations() > 0) {

          // Get a free send slot.
          int free_send_slot = free_slots_for_sending_.back();
          free_slots_for_sending_.pop_back();

          // Prepare the initial subtable to send.
          messages_to_send_[
            free_send_slot ].get<0>().Init(world, route_request);

          // Add to the list of occupied slots.
          sending_in_progress_.push_back(free_send_slot);

          // Compute the next destination to send and issue the
          // asynchronous send.
          int next_destination =
            messages_to_send_[free_send_slot].get<0>().next_destination(world);
          if(next_destination >= 0) {
            messages_to_send_[free_send_slot].get<1>() =
              world.isend(
                next_destination,
                core::parallel::MessageTag::FINISHED_TUPLES,
                messages_to_send_[free_send_slot].get<0>());
          }
        }

        // Pop it from the list.
        queued_up_completed_computation_.pop_back();
      }

      // Check whether the send requests in progress can be advanced
      // to the next state. If so, keep issuing asynchronous send.
      for(int i = 0; i < static_cast<int>(sending_in_progress_.size()); i++) {
        int send_subtable_to_test = sending_in_progress_[i];
        boost::tuple < RouteRequestType,
              boost::mpi::request > &route_request =
                messages_to_send_[ send_subtable_to_test ];
        if(route_request.get<1>().test()) {

          // Compute the next destination to send.
          int next_destination = -1;
          if(route_request.get<0>().num_destinations() > 0) {
            next_destination =
              route_request.get<0>().next_destination(world);
          }

          // If the next destination is valid, then issue another
          // asynchronous send.
          if(next_destination >= 0) {
            route_request.get<1>() =
              world.isend(
                next_destination,
                core::parallel::MessageTag::FINISHED_TUPLES,
                route_request.get<0>());
          }

          // Otherwise, free up the send slot, if no more destinations
          // are left.
          else if(route_request.get<0>().num_destinations() == 0) {
            sending_in_progress_[i] = sending_in_progress_.back();
            sending_in_progress_.pop_back();
            free_slots_for_sending_.push_back(send_subtable_to_test);

            // Decrement so that the current index can be re-tested.
            i--;
          }
        } // end of testing whether the send is complete.
      }

      // Queue incoming receives as long as there is a free slot.
      for(int i = 0; i < static_cast<int>(log2(world.size())) &&
          free_slots_for_receiving_.size() > 0; i++) {

        // Probe whether there is an incoming reference subtable.
        int source = world.rank() ^(1 << i);
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                source,
                core::parallel::MessageTag::FINISHED_TUPLES)) {

          // Get a free cache block.
          int free_receive_slot = free_slots_for_receiving_.back();
          free_slots_for_receiving_.pop_back();

          // Prepare the subtable to be received.
          std::pair < RouteRequestType,
              boost::mpi::request > &route_request_pair =
                messages_to_receive_[ free_receive_slot ];
          RouteRequestType &route_request = route_request_pair.first;
          world.recv(
            source,
            core::parallel::MessageTag::FINISHED_TUPLES,
            route_request);

          // Decrement the remaining work.
          if(route_request.remove_from_destination_list(world.rank())) {
            remaining_computation_ -= route_request.object();
          }

          // If the messsage is empty, then return the slot.
          if(route_request.num_destinations() == 0) {
            free_slots_for_receiving_.push_back(free_receive_slot);
            continue;
          }

          // Queue up for sending from the receive slot, if there are
          // additional destinations.
          if(route_request.num_destinations() > 0) {

            // Lock the received reference subtable equal to the
            // number of additional destinations.
            this->LockCache(
              free_receive_slot, route_request.num_destinations());
            route_request_pair.second =
              world.isend(
                route_request.next_destination(world),
                core::parallel::MessageTag::FINISHED_TUPLES,
                route_request) ;
            queue_up_for_sending_from_receive_slots_.push_back(
              free_receive_slot);
          }
        }
      }
    }
};
}
}

#endif
