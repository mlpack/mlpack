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
    typedef core::parallel::RouteRequest< std::pair<int, unsigned long int> > MessageRouteRequestType;

    typedef core::parallel::RouteRequest<int> SynchRouteRequestType;

  private:

    unsigned long int remaining_computation_;

    unsigned long int local_work_;

    unsigned int max_stage_;

    std::vector <
    std::pair <
    MessageRouteRequestType, boost::mpi::request > > message_cache_;

    std::vector<int> message_sending_in_progress_;

    std::vector<MessageRouteRequestType> queued_up_completed_computation_;

    int reached_max_stage_count_;

    unsigned int stage_;

    std::vector <
    std::pair <
    SynchRouteRequestType, boost::mpi::request > > synch_messages_;

    std::vector<int> synch_message_sending_in_progress_;

  private:

    void AsynchBarrier_(boost::mpi::communicator &world) {

      // Test whether the outgoing sends are complete.
      for(int i = 0;
          i < static_cast<int>(synch_message_sending_in_progress_.size()); i++) {

        int send_test_index = synch_message_sending_in_progress_[i];
        std::pair < SynchRouteRequestType,
            boost::mpi::request > &route_request_pair =
              synch_messages_[ send_test_index ];
        if(route_request_pair.second.test()) {

          // If more destinations left, then re-issue. Otherwise free.
          if(route_request_pair.first.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request_pair.first,
              core::parallel::MessageTag::SYNCHRONIZE_IN_TERMINATION,
              send_test_index,
              (std::vector<int> *) NULL);
          }
          else {
            synch_message_sending_in_progress_[i] =
              synch_message_sending_in_progress_.back();
            synch_message_sending_in_progress_.pop_back();
            i--;
          }
        }
      }

      // Send and receive from the partner of all log_2 P neighbors by
      // IProbing.
      for(unsigned int i = 0; i < max_stage_; i++) {
        int neighbor = world.rank() ^ (1 << i);
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::SYNCHRONIZE_IN_TERMINATION)) {

          // Receive the message and increment the count.
          SynchRouteRequestType tmp_route_request;
          world.recv(
            neighbor,
            core::parallel::MessageTag::SYNCHRONIZE_IN_TERMINATION,
            tmp_route_request);
          int cache_id = tmp_route_request.object();
          std::pair < SynchRouteRequestType,
              boost::mpi::request > &route_request_pair =
                synch_messages_[ cache_id ];
          route_request_pair.first = tmp_route_request;
          reached_max_stage_count_++;

          // Remove self.
          route_request_pair.first.remove_from_destination_list(world.rank());

          // Forward the barrier message.
          if(route_request_pair.first.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request_pair.first,
              core::parallel::MessageTag::SYNCHRONIZE_IN_TERMINATION,
              cache_id,
              &synch_message_sending_in_progress_);
          }
        }
      }
    }

    template<typename ObjectType>
    void IssueSending_(
      boost::mpi::communicator &world,
      boost::mpi::request &request,
      ObjectType &object,
      enum core::parallel::MessageTag::MessageTagType message_tag,
      int object_id,
      std::vector<int> *sending_in_progress_in) {


      // If this is a new send, then update the list and lock the
      // cache.
      if(sending_in_progress_in != NULL) {
        sending_in_progress_in->push_back(object_id);
      }

      request = world.isend(
                  object.next_destination(world),
                  message_tag, object);
    }

  public:

    void push_completed_computation(
      boost::mpi::communicator &comm, unsigned long int quantity_in) {

      // Subtract from the self and queue up a route message so that
      // it can be passed to all the other processes.
      remaining_computation_ -= quantity_in;
      local_work_ -= quantity_in;
      printf("Process %d left: %d\n", comm.rank(), local_work_);
      if(comm.size() > 1) {
        if(queued_up_completed_computation_.size() == 0) {
          MessageRouteRequestType new_route_request;
          new_route_request.Init(comm);
          new_route_request.set_object_is_valid_flag(true);
          new_route_request.object() =
            std::pair<int, unsigned long int>(comm.rank(), quantity_in);
          new_route_request.add_destinations(comm);
          queued_up_completed_computation_.push_back(new_route_request);
        }
        else {
          queued_up_completed_computation_.back().object().second =
            queued_up_completed_computation_.back().object().second +
            quantity_in;
        }
      }
    }

    bool can_terminate(boost::mpi::communicator &world) const {

      // Safely terminate when the current process has confirmed that
      // all computation has been completed.
      if(remaining_computation_ == 0 &&
          queued_up_completed_computation_.size() == 0 &&
          message_sending_in_progress_.size() == 0 &&
          synch_message_sending_in_progress_.size() == 0) {
      }
      return remaining_computation_ == 0 &&
             queued_up_completed_computation_.size() == 0 &&
             message_sending_in_progress_.size() == 0 &&
             synch_message_sending_in_progress_.size() == 0;
    }

    DistributedTermination() {
      local_work_ = 0;
      max_stage_ = 0;
      reached_max_stage_count_ = 0;
      remaining_computation_ = 0;
      stage_ = 0;
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

      // Preallocate the message cache.
      message_cache_.resize(world.size());
      message_sending_in_progress_.resize(0);
      synch_messages_.resize(world.size());
      synch_message_sending_in_progress_.resize(0);

      // Used for synchronizing at the end of each phase.
      reached_max_stage_count_ = world.size();

      // Initialize the stage.
      stage_ = 0;

      // The maximum number of neighbors.
      max_stage_ = static_cast<unsigned int>(log2(world.size()));
    }

    void AsynchForwardTerminationMessages(boost::mpi::communicator &world) {

      // Nothing to do, if alone.
      if(world.size() == 1) {
        return;
      }
      return;

      // At the start of each phase (stage == 0), dequeue something
      // from the hashed list.
      if(stage_ == 0) {

        // Wait for others before starting the stage.
        if(reached_max_stage_count_ < world.size() ||
            synch_message_sending_in_progress_.size() > 0) {
          AsynchBarrier_(world);
          return;
        }

        // Reset the count once we have begun the 0-th stage.
        reached_max_stage_count_ = 0;

        // The status and the object to be copied onto.
        boost::mpi::request &new_self_send_request =
          message_cache_[ world.rank()].second;
        MessageRouteRequestType &new_self_send_request_object =
          message_cache_[ world.rank()].first;
        if(queued_up_completed_computation_.size() > 0) {

          // Examine the back of the route request list.
          MessageRouteRequestType &route_request =
            queued_up_completed_computation_.back();

          // Prepare the initial subtable to send.
          new_self_send_request_object.Init(world, route_request);

          // Pop it from the route request list.
          queued_up_completed_computation_.pop_back();
        }
        else {

          // Prepare an empty message.
          new_self_send_request_object.Init(world);
          new_self_send_request_object.object() =
            std::pair<int, unsigned long int>(world.rank(), 0);
          new_self_send_request_object.set_object_is_valid_flag(true);
          new_self_send_request_object.add_destinations(world);
        }

        // Issue an asynchronous send.
        IssueSending_(
          world, new_self_send_request,
          new_self_send_request_object,
          core::parallel::MessageTag::FINISHED_TUPLES,
          world.rank(),
          &message_sending_in_progress_);

        // Increment the stage.
        stage_++;
      }

      // Test whether the send issues are complete.
      for(int i = 0;
          i < static_cast<int>(message_sending_in_progress_.size()); i++) {

        int send_test_index = message_sending_in_progress_[i];
        std::pair < MessageRouteRequestType,
            boost::mpi::request > &route_request_pair =
              message_cache_[ send_test_index ];
        if(route_request_pair.second.test()) {

          // If more destinations left, then re-issue. Otherwise free.
          if(route_request_pair.first.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request_pair.first,
              core::parallel::MessageTag::FINISHED_TUPLES,
              send_test_index,
              (std::vector<int> *) NULL);
          }
          else {
            message_sending_in_progress_[i] =
              message_sending_in_progress_.back();
            message_sending_in_progress_.pop_back();
            i--;
          }
        }
      }

      // Send and receive from the partner of all log_2 P neighbors by
      // IProbing.
      for(unsigned int i = 0; i < max_stage_; i++) {
        int neighbor = world.rank() ^ (1 << i);
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::FINISHED_TUPLES)) {

          // Receive the subtable and increment the count.
          MessageRouteRequestType tmp_route_request;
          world.recv(
            neighbor,
            core::parallel::MessageTag::FINISHED_TUPLES,
            tmp_route_request);
          int cache_id = tmp_route_request.object().first;
          std::pair < MessageRouteRequestType,
              boost::mpi::request > &route_request_pair =
                message_cache_[ cache_id ];
          route_request_pair.first = tmp_route_request;
          stage_++;

          // If this subtable is needed by the calling process, then
          // update the list of subtables received.
          if(route_request_pair.first.remove_from_destination_list(world.rank()) &&
              route_request_pair.first.object_is_valid()) {
            remaining_computation_ -= route_request_pair.first.object().second;
          }

          // If there are more destinations left for the received
          // subtable, lock the cache appropriately and issue and
          // asynchronous send.
          if(route_request_pair.first.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request_pair.first,
              core::parallel::MessageTag::FINISHED_TUPLES,
              neighbor,
              &message_sending_in_progress_);
          }
        } // end of receiving a message.
      }

      // If at the end of phase, wait for others to reach this point.
      if(stage_ == message_cache_.size() &&
          message_sending_in_progress_.size() == 0) {
        stage_ = 0;
        reached_max_stage_count_++;
        std::pair < SynchRouteRequestType,
            boost::mpi::request > &synch_request_pair =
              synch_messages_[ world.rank()];
        SynchRouteRequestType &synch_request = synch_request_pair.first;

        // Reset the message and send.
        synch_request.Init(world);
        synch_request.add_destinations(world);
        synch_request.object() = world.rank();
        synch_request.set_object_is_valid_flag(true);
        IssueSending_(
          world, synch_request_pair.second, synch_request,
          core::parallel::MessageTag::SYNCHRONIZE_IN_TERMINATION, world.rank(),
          &synch_message_sending_in_progress_);
      }
    }
};
}
}

#endif
