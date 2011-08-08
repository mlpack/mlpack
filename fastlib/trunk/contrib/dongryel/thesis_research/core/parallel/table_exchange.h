/** @file table_exchange.h
 *
 *  A class to do a set of all-to-some table exchanges asynchronously.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_H

#include <boost/mpi.hpp>
#include "core/parallel/message_tag.h"
#include "core/parallel/route_request.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/dense_matrix.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

/** @brief A class for performing an all-to-some exchange of subtrees
 *         among MPI processes.
 */
template<typename DistributedTableType>
class TableExchange {
  public:

    /** @brief The table type used in the exchange process.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The subtable type used in the exchange process.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The old from new index type used in the exchange
     *         process.
     */
    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    typedef core::parallel::RouteRequest<SubTableType> SubTableRouteRequestType;

    typedef core::parallel::RouteRequest<bool> SynchRouteRequestType;

  private:

    /** @brief The pointer to the local table that is partcipating in
     *         the exchange.
     */
    TableType *local_table_;

    int max_stage_;

    int reached_max_stage_count_;

    unsigned int stage_;

    std::vector <
    std::pair <
    SubTableRouteRequestType, boost::mpi::request > > subtable_cache_;

    std::vector< int > subtable_sending_in_progress_;

    std::vector <
    std::pair <
    SynchRouteRequestType, boost::mpi::request > > synch_messages_;

    std::vector<int> synch_message_sending_in_progress_;

    std::vector< int > subtable_locks_;

    int total_num_locks_;

  private:

    /** @brief Prints the existing subtables in the cache.
     */
    void PrintSubTables_(boost::mpi::communicator &world) {
      printf("\n\nProcess %d owns the subtables:\n", world.rank());
      for(unsigned int i = 0; i < subtable_cache_.size(); i++) {
        printf(
          "%d %d %d\n",
          subtable_cache_[i].first.table()->rank(),
          subtable_cache_[i].first.table()->get_tree()->begin(),
          subtable_cache_[i].first.table()->get_tree()->count());
      }
    }

    void AsynchBarrier_(boost::mpi::communicator &world) {

      // Test whether the outgoing sends are complete.
      for(int i = 0;
          i < static_cast<int>(synch_message_sending_in_progress_.size()); i++) {

        int send_test_index = synch_message_sending_in_progress_[i];
        std::pair < SynchRouteRequestType,
            boost::mpi::request > &route_request_pair =
              synch_messages_[ send_test_index ];
        if(route_request_pair.second.test()) {

          // Release the cache number of times equal to the number of
          // destinations routed.
          this->ReleaseCache(
            send_test_index, route_request_pair.first.num_routed(), false);

          // If more destinations left, then re-issue. Otherwise free.
          if(route_request_pair.first.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request_pair.first,
              core::parallel::MessageTag::SYNCHRONIZE,
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
      for(int i = 0; i < max_stage_; i++) {
        int neighbor = world.rank() ^(1 << i);
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::SYNCHRONIZE)) {

          // Receive the message and increment the count.
          std::pair < SynchRouteRequestType,
              boost::mpi::request > &route_request_pair =
                synch_messages_[ neighbor ];
          SynchRouteRequestType &route_request = route_request_pair.first;
          world.recv(
            neighbor,
            core::parallel::MessageTag::SYNCHRONIZE,
            route_request);
          reached_max_stage_count_++;

          // Remove self.
          route_request.remove_from_destination_list(world.rank());

          // Forward the barrier message.
          if(route_request.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request,
              core::parallel::MessageTag::SYNCHRONIZE,
              neighbor,
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

        // Lock the cache.
        this->LockCache(
          object_id, object.num_destinations());
      }

      request = world.isend(
                  object.next_destination(world),
                  message_tag, object);
    }

  public:

    /** @brief The default constructor.
     */
    TableExchange() {
      local_table_ = NULL;
      reached_max_stage_count_ = 0;
      total_num_locks_ = 0;
    }

    /** @brief Tests whether the cache is empty or not.
     */
    bool is_empty() const {
      return total_num_locks_ == 0;
    }

    void LockCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        subtable_locks_[ cache_id ] += num_times;
        total_num_locks_ += num_times;
      }
    }

    void ReleaseCache(int cache_id, int num_times, bool free_subtable = true) {
      if(cache_id >= 0) {
        subtable_locks_[ cache_id ] -= num_times;
        total_num_locks_ -= num_times;

        // If the subtable is not needed, free it.
        if(subtable_locks_[ cache_id ] == 0 && free_subtable &&
            subtable_cache_[ cache_id ].first.object_is_valid()) {

          // This is a hack. See the assignment operator for SubTable.
          SubTableType safe_free =
            subtable_cache_[cache_id].first.object();
        }
      }
    }

    /** @brief Grabs the subtable in the given cache position.
     */
    SubTableType *FindSubTable(int cache_id) {
      SubTableType *returned_subtable = NULL;
      if(cache_id >= 0) {
        returned_subtable =
          &(subtable_cache_[cache_id].first.object());
      }
      return returned_subtable;
    }

    /** @brief Initialize the all-to-some exchange object with a
     *         distributed table and the cache size.
     */
    void Init(
      boost::mpi::communicator &world,
      TableType &local_table_in,
      int max_num_work_to_dequeue_per_stage_in) {

      // Initialize the stage.
      stage_ = 0;

      // The maximum number of neighbors.
      max_stage_ = static_cast<int>(log2(world.size()));

      // Set the local table.
      local_table_ = &local_table_in;

      // Preallocate the cache.
      subtable_cache_.resize(world.size());
      subtable_sending_in_progress_.resize(0);
      synch_messages_.resize(world.size());
      synch_message_sending_in_progress_.resize(0);

      // Initialize the locks.
      subtable_locks_.resize(subtable_cache_.size());
      std::fill(
        subtable_locks_.begin(), subtable_locks_.end(), 0);
      total_num_locks_ = 0;

      // Used for synchronizing at the end of each phase.
      reached_max_stage_count_ = world.size();
    }

    /** @brief Issue a set of asynchronous send and receive
     *         operations.
     *
     *  @return received_subtables The list of received subtables.
     */
    void AsynchSendReceive(
      boost::mpi::communicator &world,
      std::vector <
      SubTableRouteRequestType > &hashed_essential_reference_subtrees_to_send,
      std::vector< boost::tuple<int, int, int, int> > *received_subtable_ids,
      int *num_completed_sends) {

      // If the number of processes is only one, then don't bother
      // since there is nothing to exchange.
      if(world.size() == 1) {
        return;
      }

      // Clear the list of received subtables in this round.
      received_subtable_ids->resize(0);

      // At the start of each phase (stage == 0), dequeue something
      // from the hashed list.
      if(stage_ == 0) {

        // Wait for others before starting the stage.
        if(reached_max_stage_count_ < world.size() ||
            synch_message_sending_in_progress_.size() > 0 ||
            total_num_locks_ > 0) {
          AsynchBarrier_(world);
          return;
        }

        // Reset the count once we have begun the 0-th stage.
        reached_max_stage_count_ = 0;

        // The status and the object to be copied onto.
        boost::mpi::request &new_self_send_request =
          subtable_cache_[ world.rank()].second;
        SubTableRouteRequestType &new_self_send_request_object =
          subtable_cache_[ world.rank()].first;
        if(hashed_essential_reference_subtrees_to_send.size() > 0) {

          // Examine the back of the route request list.
          SubTableRouteRequestType &route_request =
            hashed_essential_reference_subtrees_to_send.back();

          // Prepare the initial subtable to send.
          new_self_send_request_object.Init(world, route_request);

          // Pop it from the route request list.
          hashed_essential_reference_subtrees_to_send.pop_back();
        }
        else {

          // Prepare an empty message.
          new_self_send_request_object.Init(world);
          new_self_send_request_object.add_destinations(world);
        }

        // Issue an asynchronous send.
        IssueSending_(
          world, new_self_send_request,
          new_self_send_request_object,
          core::parallel::MessageTag::ROUTE_SUBTABLE,
          world.rank(),
          &subtable_sending_in_progress_);

        // Increment the stage.
        stage_++;
      }

      // Test whether the send issues are complete.
      for(int i = 0;
          i < static_cast<int>(subtable_sending_in_progress_.size()); i++) {

        int send_test_index = subtable_sending_in_progress_[i];
        std::pair < SubTableRouteRequestType,
            boost::mpi::request > &route_request_pair =
              subtable_cache_[ send_test_index ];
        if(route_request_pair.second.test()) {

          // Release cache equal to the number of destinations routed.
          this->ReleaseCache(
            send_test_index, route_request_pair.first.num_routed());

          // If more destinations left, then re-issue. Otherwise free.
          if(route_request_pair.first.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request_pair.first,
              core::parallel::MessageTag::ROUTE_SUBTABLE,
              send_test_index,
              (std::vector<int> *) NULL);
          }
          else {
            subtable_sending_in_progress_[i] =
              subtable_sending_in_progress_.back();
            subtable_sending_in_progress_.pop_back();
            i--;
          }
        }
      }

      // Send and receive from the partner of all log_2 P neighbors by
      // IProbing.
      for(int i = 0; i < max_stage_; i++) {
        int neighbor = world.rank() ^(1 << i);
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(
                neighbor,
                core::parallel::MessageTag::ROUTE_SUBTABLE)) {

          // Receive the subtable and increment the count.
          std::pair < SubTableRouteRequestType,
              boost::mpi::request > &route_request_pair =
                subtable_cache_[ neighbor ];
          SubTableRouteRequestType &route_request = route_request_pair.first;
          route_request.object().Init(neighbor, false);
          world.recv(
            neighbor,
            core::parallel::MessageTag::ROUTE_SUBTABLE,
            route_request);
          stage_++;

          // If this subtable is needed by the calling process, then
          // update the list of subtables received.
          if(route_request.remove_from_destination_list(world.rank()) &&
              route_request.object_is_valid()) {
            received_subtable_ids->push_back(
              boost::make_tuple(
                route_request.object().table()->rank(),
                route_request.object().start_node()->begin(),
                route_request.object().start_node()->count(),
                neighbor));
          }

          // If there are more destinations left for the received
          // subtable, lock the cache appropriately and issue and
          // asynchronous send.
          if(route_request.num_destinations() > 0) {
            IssueSending_(
              world, route_request_pair.second,
              route_request,
              core::parallel::MessageTag::ROUTE_SUBTABLE,
              neighbor,
              &subtable_sending_in_progress_);
          }
        } // end of receiving a message.
      }

      // If at the end of phase, wait for others to reach this point.
      if(stage_ == subtable_cache_.size() &&
          subtable_sending_in_progress_.size() == 0 &&
          total_num_locks_ == 0) {
        stage_ = 0;
        reached_max_stage_count_++;
        std::pair < SynchRouteRequestType,
            boost::mpi::request > &synch_request_pair =
              synch_messages_[ world.rank()];
        SynchRouteRequestType &synch_request = synch_request_pair.first;

        // Reset the message and send.
        synch_request.Init(world);
        synch_request.add_destinations(world);
        IssueSending_(
          world, synch_request_pair.second, synch_request,
          core::parallel::MessageTag::SYNCHRONIZE, world.rank(),
          &synch_message_sending_in_progress_);
      }
    }
};
}
}

#endif
