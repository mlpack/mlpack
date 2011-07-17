/** @file table_exchange.h
 *
 *  A class to do a set of all-to-all table exchanges.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_H

#include <boost/mpi.hpp>
#include "core/table/memory_mapped_file.h"
#include "core/table/dense_matrix.h"
#include "core/table/sub_table.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace parallel {

/** @brief A class for performing an all-to-all exchange of subtrees
 *         among MPI processes.
 */
template<typename DistributedTableType, typename SubTableListType>
class TableExchange {
  public:

    /** @brief The table type used in the exchange process.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The subtable type used in the exchange process.
     */
    typedef typename SubTableListType::SubTableType SubTableType;

    /** @brief The old from new index type used in the exchange
     *         process.
     */
    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

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
    SubTableType, boost::mpi::request > > subtables_to_receive_;

    std::vector< int > receive_cache_locks_;

    std::vector<int> receiving_in_progress_;

    std::vector<int> free_slots_for_receiving_;

    std::vector <
    boost::tuple <
    SubTableType, boost::mpi::request, int > > subtables_to_send_;

    std::vector<int> sending_in_progress_;

    std::vector<int> free_slots_for_sending_;

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

    bool is_empty() const {
      return
        static_cast<int>(free_slots_for_receiving_.size()) ==
        num_cache_blocks_ &&
        static_cast<int>(free_slots_for_sending_.size()) == num_cache_blocks_;
    }

    void LockCache(int cache_id, int num_times) {
      if(cache_id >= 0) {
        receive_cache_locks_[ cache_id ] += num_times;
      }
    }

    void ReleaseCache(int cache_id) {
      if(cache_id >= 0) {
        receive_cache_locks_[ cache_id ] --;

        // If the subtable is not needed, feel free to free it.
        if(receive_cache_locks_[ cache_id ] == 0) {

          // This is a hack. See the assignment operator for SubTable.
          SubTableType safe_free = subtables_to_receive_[cache_id].first;

          free_slots_for_receiving_.push_back(cache_id);
        }
      }
    }

    /** @brief Grabs the subtable in the given cache position.
     */
    SubTableType *FindSubTable(int cache_id) {
      SubTableType *returned_subtable = NULL;
      if(cache_id >= 0) {
        returned_subtable = &(subtables_to_receive_[cache_id].first);
      }
      return returned_subtable;
    }

    /** @brief Initialize the all-to-all exchange object with a
     *         distributed table and the cache size.
     */
    void Init(
      boost::mpi::communicator &world,
      TableType &local_table_in,
      int max_num_work_to_dequeue_per_stage_in) {

      // Set the local table.
      local_table_ = &local_table_in;

      // Compute the number of cache blocks allocated per process.
      num_cache_blocks_ = max_num_work_to_dequeue_per_stage_in + 1;

      if(world.rank() == 0) {
        printf(
          "Number of cache blocks per process: %d\n",
          num_cache_blocks_);
      }

      // Preallocate the send and receive caches.
      subtables_to_receive_.resize(num_cache_blocks_);
      subtables_to_send_.resize(num_cache_blocks_);

      // Initialize the locks.
      receive_cache_locks_.resize(num_cache_blocks_);
      std::fill(
        receive_cache_locks_.begin(), receive_cache_locks_.end(), 0);

      // Allocate the free cache block lists.
      for(int j = 0; j < num_cache_blocks_; j++) {
        free_slots_for_receiving_.push_back(j);
        free_slots_for_sending_.push_back(j);
      }
      printf(
        "Process %d finished initializing the cache blocks.\n", world.rank());
    }

    /** @brief Issue a set of asynchronous send and receive
     *         operations.
     *
     *  @return received_subtables The list of received subtables.
     */
    template<typename SendRequestPriorityQueueType>
    void AsynchSendReceive(
      boost::mpi::communicator &world,
      std::vector< SendRequestPriorityQueueType > &prioritized_send_subtables,
      std::vector< boost::tuple<int, int, int, int> > *received_subtable_ids,
      int *num_completed_sends) {

      // Clear the list of received subtables in this round.
      received_subtable_ids->resize(0);

      // Initiate send requests.
      while(free_slots_for_sending_.size() > 0) {

        // This is so that we do not send everything to one single
        // process, i.e. pseudo-load-balancing.
        int probe_start = core::math::RandInt(world.size());
        bool all_empty = true;

        for(int i = 0; all_empty && i < world.size(); i++) {

          // Skip empty lists.
          int probe_index = (i + probe_start) % world.size();
          if(prioritized_send_subtables[probe_index].size() == 0) {
            continue;
          }
          all_empty = false;

          // Take a peek at the top send request and pop it.
          const typename
          SendRequestPriorityQueueType::value_type &top_send_request =
            prioritized_send_subtables[probe_index].top();
          int destination = top_send_request.destination();
          int begin = top_send_request.begin();
          int count = top_send_request.count();
          prioritized_send_subtables[probe_index].pop();

          // Get a free send slot.
          int free_send_slot = free_slots_for_sending_.back();
          free_slots_for_sending_.pop_back();

          // Prepare the subtable to send.
          subtables_to_send_[
            free_send_slot].get<0>().Init(
              local_table_,
              local_table_->get_tree()->FindByBeginCount(begin, count),
              false);

          // Issue an asynchronous send and break.
          subtables_to_send_[free_send_slot].get<1>() =
            world.isend(
              destination, begin, subtables_to_send_[free_send_slot].get<0>());
          subtables_to_send_[free_send_slot].get<2>() = destination;
          sending_in_progress_.push_back(free_send_slot);
          break;
        }

        // If nothing was found, then break.
        if(all_empty) {
          break;
        }
      } // end of queuing up the sends.

      // Queue incoming receives as long as there is a free slot.
      while(free_slots_for_receiving_.size() > 0) {

        // Probe whether there is an incoming reference subtable.
        if(boost::optional< boost::mpi::status > l_status =
              world.iprobe(boost::mpi::any_source, boost::mpi::any_tag)) {

          // Get a free cache block.
          int free_receive_slot = free_slots_for_receiving_.back();
          free_slots_for_receiving_.pop_back();

          // Prepare the subtable to be received.
          subtables_to_receive_[ free_receive_slot ].first.Init(
            free_receive_slot, false);
          world.recv(
            l_status->source(),
            l_status->tag(),
            subtables_to_receive_[ free_receive_slot ].first);
          receiving_in_progress_.push_back(free_receive_slot);

          received_subtable_ids->push_back(
            boost::make_tuple(
              subtables_to_receive_[
                free_receive_slot ].first.table()->rank(),
              subtables_to_receive_[
                free_receive_slot ].first.table()->get_tree()->begin(),
              subtables_to_receive_[
                free_receive_slot ].first.table()->get_tree()->count(),
              free_receive_slot));
        }
        else {
          break;
        }
      }

      // Check whether the subtables have been sent.
      for(int i = 0; i < static_cast<int>(sending_in_progress_.size()); i++) {
        int send_subtable_to_test = sending_in_progress_[i];
        if(subtables_to_send_[ send_subtable_to_test ].get<1>().test()) {

          // Free up the send slot.
          sending_in_progress_[i] = sending_in_progress_.back();
          sending_in_progress_.pop_back();
          free_slots_for_sending_.push_back(send_subtable_to_test);

          // Tally the finished sends.
          (*num_completed_sends)++;

          // Decrement so that the current index can be re-tested.
          i--;
        }
      }
    }
};
}
}

#endif
