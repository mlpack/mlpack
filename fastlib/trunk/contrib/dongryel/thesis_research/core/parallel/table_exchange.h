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

    typedef core::parallel::RouteRequest<SubTableType> RouteRequestType;

  private:

    /** @brief The pointer to the local table that is partcipating in
     *         the exchange.
     */
    TableType *local_table_;

    /** @brief The number of cache blocks per process.
     */
    int num_cache_blocks_;

    std::vector <
    std::pair <
    RouteRequestType, boost::mpi::request > > subtables_to_receive_;

    std::vector< int > receive_cache_locks_;

    std::vector<int> free_slots_for_receiving_;

    int total_num_subtables_received_;

    std::vector <
    boost::tuple <
    RouteRequestType, boost::mpi::request > > subtables_to_send_;

    std::vector<int> sending_in_progress_;

    std::vector<int> free_slots_for_sending_;

    std::vector<int> queue_up_for_sending_from_receive_slots_;

  private:

    /** @brief Prints the existing subtables in the cache.
     */
    void PrintSubTables_(boost::mpi::communicator &world) {
      printf("\n\nProcess %d owns the subtables:\n", world.rank());
      for(unsigned int i = 0; i < subtables_to_receive_.size(); i++) {
        printf(
          "%d %d %d\n",
          subtables_to_receive_[i].first.table()->rank(),
          subtables_to_receive_[i].first.table()->get_tree()->begin(),
          subtables_to_receive_[i].first.table()->get_tree()->count());
      }
    }

  public:

    int total_num_subtables_received() const {
      return total_num_subtables_received_;
    }

    /** @brief The default constructor.
     */
    TableExchange() {
      local_table_ = NULL;
      num_cache_blocks_ = 0;
      total_num_subtables_received_ = 0;
    }

    /** @brief Tests whether the cache is empty or not.
     */
    bool is_empty() const {
      return
        free_slots_for_receiving_.size() == subtables_to_receive_.size() &&
        free_slots_for_sending_.size() == subtables_to_send_.size();
    }

    void LockCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        receive_cache_locks_[ cache_id ] += num_times;
      }
    }

    void ReleaseCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        receive_cache_locks_[ cache_id ] -= num_times;

        // If the subtable is not needed, feel free to free it.
        if(receive_cache_locks_[ cache_id ] == 0) {

          // This is a hack. See the assignment operator for SubTable.
          SubTableType safe_free =
            subtables_to_receive_[cache_id].first.object();
          free_slots_for_receiving_.push_back(cache_id);
        }
      }
    }

    /** @brief Grabs the subtable in the given cache position.
     */
    SubTableType *FindSubTable(int cache_id) {
      SubTableType *returned_subtable = NULL;
      if(cache_id >= 0) {
        returned_subtable =
          &(subtables_to_receive_[cache_id].first.object());
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

      // Set the local table.
      local_table_ = &local_table_in;

      // Compute the number of cache blocks allocated per process.
      num_cache_blocks_ = max_num_work_to_dequeue_per_stage_in;
      num_cache_blocks_ = 1;

      if(world.rank() == 0) {
        printf(
          "Number of cache blocks per process: %d\n",
          num_cache_blocks_);
      }

      // Preallocate the send and receive caches.
      subtables_to_receive_.resize(num_cache_blocks_ * world.size());
      subtables_to_send_.resize(num_cache_blocks_);
      for(unsigned int i = 0; i < subtables_to_send_.size(); i++) {
        free_slots_for_sending_.push_back(i);
      }
      for(unsigned int i = 0; i < subtables_to_receive_.size(); i++) {
        free_slots_for_receiving_.push_back(i);
      }

      // Initialize the locks.
      receive_cache_locks_.resize(free_slots_for_receiving_.size());
      std::fill(
        receive_cache_locks_.begin(), receive_cache_locks_.end(), 0);

      printf(
        "Process %d finished initializing the cache blocks.\n", world.rank());
    }

    /** @brief Issue a set of asynchronous send and receive
     *         operations.
     *
     *  @return received_subtables The list of received subtables.
     */
    void AsynchSendReceive(
      boost::mpi::communicator &world,
      std::vector <
      RouteRequestType > &hashed_essential_reference_subtrees_to_send,
      std::vector< boost::tuple<int, int, int, int> > *received_subtable_ids,
      int *num_completed_sends) {

      // If the number of processes is only one, then don't bother
      // since there is nothing to exchange.
      if(world.size() == 1) {
        return;
      }

      // Clear the list of received subtables in this round.
      received_subtable_ids->resize(0);

      // Queue up send requests by transfering from the receive route
      // requests.
      for(int i = 0; i < static_cast<int>(
            queue_up_for_sending_from_receive_slots_.size()) ; i++) {

        // Examine the back of the route request list.
        int cache_id = queue_up_for_sending_from_receive_slots_[i];
        std::pair <
        RouteRequestType, boost::mpi::request > &route_request_pair =
          subtables_to_receive_[ cache_id ];
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
                core::parallel::MessageTag::ROUTE_SUBTABLE,
                route_request) ;
          }
        }
      }

      // Queue up send requests by dequeuing from the list of
      // reference subtrees to send from the current process.
      while(free_slots_for_sending_.size() > 0 &&
            hashed_essential_reference_subtrees_to_send.size() > 0) {

        // Examine the back of the route request list.
        RouteRequestType &route_request =
          hashed_essential_reference_subtrees_to_send.back();

        // If the next destination is valid, then get a free slot and
        // issue an asynchronous send.
        if(route_request.num_destinations() > 0) {

          // Get a free send slot.
          int free_send_slot = free_slots_for_sending_.back();
          free_slots_for_sending_.pop_back();

          // Prepare the initial subtable to send.
          subtables_to_send_[
            free_send_slot ].get<0>().Init(world, route_request);

          // Add to the list of occupied slots.
          sending_in_progress_.push_back(free_send_slot);

          // Compute the next destination to send and issue the
          // asynchronous send.
          int next_destination =
            subtables_to_send_[free_send_slot].get<0>().next_destination(world);
          if(next_destination >= 0) {
            subtables_to_send_[free_send_slot].get<1>() =
              world.isend(
                next_destination,
                core::parallel::MessageTag::ROUTE_SUBTABLE,
                subtables_to_send_[free_send_slot].get<0>());
          }
        }

        // Pop it from the list.
        hashed_essential_reference_subtrees_to_send.pop_back();
      }

      // Check whether the send requests in progress can be advanced
      // to the next state. If so, keep issuing asynchronous send.
      for(int i = 0; i < static_cast<int>(sending_in_progress_.size()); i++) {
        int send_subtable_to_test = sending_in_progress_[i];
        boost::tuple < RouteRequestType,
              boost::mpi::request > &route_request =
                subtables_to_send_[ send_subtable_to_test ];
        if(route_request.get<1>().test()) {

          // Only count the number of sends if it is originating from
          // the process.
          if(route_request.get<0>().object().table()->rank() == world.rank()) {
            (*num_completed_sends) += route_request.get<0>().num_routed();
          }

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
                core::parallel::MessageTag::ROUTE_SUBTABLE,
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
                core::parallel::MessageTag::ROUTE_SUBTABLE)) {

          // Get a free cache block.
          int free_receive_slot = free_slots_for_receiving_.back();
          free_slots_for_receiving_.pop_back();

          // Prepare the subtable to be received.
          std::pair < RouteRequestType,
              boost::mpi::request > &route_request_pair =
                subtables_to_receive_[ free_receive_slot ];
          RouteRequestType &route_request = route_request_pair.first;
          route_request.object().Init(free_receive_slot, false);
          world.recv(
            source,
            core::parallel::MessageTag::ROUTE_SUBTABLE,
            route_request);

          // If the messsage is empty, then return the slot.
          if(route_request.num_destinations() == 0) {
            free_slots_for_receiving_.push_back(free_receive_slot);
            continue;
          }

          // Increment the number of subtables received if the
          // destination matches the rank of the current process.
          if(route_request.remove_from_destination_list(world.rank())) {
            total_num_subtables_received_++;
            received_subtable_ids->push_back(
              boost::make_tuple(
                route_request.object().table()->rank(),
                route_request.object().start_node()->begin(),
                route_request.object().start_node()->count(),
                free_receive_slot));
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
                core::parallel::MessageTag::ROUTE_SUBTABLE,
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
