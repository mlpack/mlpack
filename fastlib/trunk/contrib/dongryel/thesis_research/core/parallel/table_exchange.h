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

    typedef core::parallel::RouteRequest<int> SynchRouteRequestType;

  private:

    std::vector<int> cleanup_list_;

    /** @brief The pointer to the local table that is partcipating in
     *         the exchange.
     */
    TableType *local_table_;

    unsigned int max_stage_;

    std::vector<int> received_subtable_list_;

    unsigned int stage_;

    std::vector < SubTableRouteRequestType > subtable_cache_;

    std::vector< boost::mpi::request > subtable_receive_request_;

    std::vector< boost::mpi::request > subtable_send_request_;

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
          subtable_cache_[i].table()->rank(),
          subtable_cache_[i].table()->get_tree()->begin(),
          subtable_cache_[i].table()->get_tree()->count());
      }
    }

    void FreeCache_() {
      for(int i = 0; i < static_cast<int>(cleanup_list_.size()); i++) {

        // This is a hack. See the assignment operator for SubTable.
        SubTableType safe_free =
          subtable_cache_[ cleanup_list_[i] ].object();
        cleanup_list_[i] = cleanup_list_.back();
        cleanup_list_.pop_back();
        i--;
      }
    }

  public:

    /** @brief The default constructor.
     */
    TableExchange() {
      local_table_ = NULL;
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
            subtable_cache_[ cache_id ].object_is_valid() &&
            cache_id != local_table_->rank()) {

          // Clean up the list later.
          cleanup_list_.push_back(cache_id);
        }
      }
    }

    /** @brief Grabs the subtable in the given cache position.
     */
    SubTableType *FindSubTable(int cache_id) {
      SubTableType *returned_subtable = NULL;
      if(cache_id >= 0) {
        returned_subtable =
          &(subtable_cache_[cache_id].object());
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

      // Clean up list is empty.
      cleanup_list_.resize(0);
      received_subtable_list_.resize(0);

      // Initialize the stage.
      stage_ = 0;

      // The maximum number of neighbors.
      max_stage_ = static_cast<unsigned int>(log2(world.size()));

      // Set the local table.
      local_table_ = &local_table_in;

      // Preallocate the cache.
      subtable_cache_.resize(world.size());
      subtable_send_request_.resize(world.size());
      subtable_receive_request_.resize(world.size());

      // Initialize the locks.
      subtable_locks_.resize(subtable_cache_.size());
      std::fill(
        subtable_locks_.begin(), subtable_locks_.end(), 0);
      total_num_locks_ = 0;
    }

    /** @brief Issue a set of asynchronous send and receive
     *         operations.
     *
     *  @return received_subtables The list of received subtables.
     */
    void SendReceive(
      boost::mpi::communicator &world,
      std::vector <
      SubTableRouteRequestType > &hashed_essential_reference_subtrees_to_send,
      int num_local_query_trees,
      std::vector< boost::tuple<int, int, int, int> > *received_subtable_ids) {

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

        // The status and the object to be copied onto.
        SubTableRouteRequestType &new_self_send_request_object =
          subtable_cache_[ world.rank()];
        if(hashed_essential_reference_subtrees_to_send.size() > 0) {

          // Examine the back of the route request list.
          SubTableRouteRequestType &route_request =
            hashed_essential_reference_subtrees_to_send.back();

          // Prepare the initial subtable to send.
          new_self_send_request_object.Init(world, route_request);
          new_self_send_request_object.set_object_is_valid_flag(true);

          // Pop it from the route request list.
          hashed_essential_reference_subtrees_to_send.pop_back();
        }
        else {

          // Prepare an empty message.
          new_self_send_request_object.Init(world);
          new_self_send_request_object.add_destinations(world);
        }
        received_subtable_list_.push_back(world.rank());
      }

      if(stage_ < max_stage_) {

        // Exchange with the neighbors.
        int num_subtables_to_exchange = (1 << stage_);
        int neighbor = world.rank() ^ (1 << stage_);
        for(int i = 0; i < num_subtables_to_exchange; i++) {
          int subtable_send_index = received_subtable_list_[i];
          boost::mpi::request &send_request =
            subtable_send_request_[ i ];
          SubTableRouteRequestType &send_request_object =
            subtable_cache_[ subtable_send_index ];

          // For each subtable sent, we expect something from the neighbor.
          send_request =
            world.isend(
              neighbor, core::parallel::MessageTag::ROUTE_SUBTABLE,
              send_request_object);
        }
        int num_subtables_received = 0;
        while(num_subtables_received < num_subtables_to_exchange) {

          if(boost::optional< boost::mpi::status > l_status =
                world.iprobe(
                  neighbor,
                  core::parallel::MessageTag::ROUTE_SUBTABLE)) {

            // Receive the subtable and increment the count.
            SubTableRouteRequestType tmp_route_request;
            tmp_route_request.object().Init(neighbor, false);
            world.recv(
              neighbor,
              core::parallel::MessageTag::ROUTE_SUBTABLE,
              tmp_route_request);
            int cache_id = tmp_route_request.object().table()->rank();
            tmp_route_request.object().set_cache_block_id(cache_id);
            subtable_cache_[ cache_id ] = tmp_route_request;
            SubTableRouteRequestType &route_request = subtable_cache_[cache_id];
            received_subtable_list_.push_back(cache_id);

            // If this subtable is needed by the calling process, then
            // update the list of subtables received.
            num_subtables_received++;
            if(route_request.remove_from_destination_list(world.rank()) &&
                route_request.object_is_valid()) {
              this->LockCache(cache_id, num_local_query_trees);
              received_subtable_ids->push_back(
                boost::make_tuple(
                  route_request.object().table()->rank(),
                  route_request.object().start_node()->begin(),
                  route_request.object().start_node()->count(),
                  cache_id));
            }
          }
        }

        // Wait until all sends are done.
        boost::mpi::wait_all(
          subtable_send_request_.begin(),
          subtable_send_request_.begin() + num_subtables_to_exchange);

        // Increment the stage when done.
        stage_++;
      }

      // If at the end of phase, wait for others to reach this point.
      else if(total_num_locks_ == 0) {

        // Reset and prepare for the next round.
        stage_ = 0;
        received_subtable_list_.resize(0);

        // Clean up the subtables.
        FreeCache_();
      }
    }
};
}
}

#endif
