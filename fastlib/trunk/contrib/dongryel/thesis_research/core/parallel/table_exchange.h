/** @file table_exchange.h
 *
 *  A class to do a set of all-to-all table exchanges.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_H

#include <boost/circular_buffer.hpp>
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

    /** @brief The cache of points received per process.
     */
    std::vector< core::table::DenseMatrix > point_cache_;

    /** @brief The cache of old from new mapping received per process.
     */
    std::vector< OldFromNewIndexType *> old_from_new_cache_;

    /** @brief The list of free cache blocks per process.
     */
    std::vector< std::vector<int> > free_cache_blocks_;

    /** @brief The number of points that could fit into a cache block.
     */
    int cache_block_size_;

    /** @brief The number of cache blocks per process.
     */
    int num_cache_blocks_per_process_;

    /** @brief The total number of points that could fit into a
     *         process cache.
     */
    int total_cache_size_per_process_;

    /** @brief The circular buffer that acts as the cache of received
     *         subtables.
     */
    std::vector< boost::circular_buffer<SubTableType> > received_subtables_;

  private:

    int get_free_cache_block_(
      int process_id, core::table::DenseMatrix *point_cache_block_alias,
      OldFromNewIndexType **old_from_new_cache_alias) {

      boost::circular_buffer<SubTableType> &circular_buffer =
        received_subtables_[ process_id ];

      int free_cache_block_id = -1;
      if(free_cache_blocks_[process_id].size() == 0) {

        // In this case, the buffer is full. If so, we need to take
        // out the head element (which is going to be overwritten) and
        // destruct it manually since Boost circular buffer does not
        // do so automatically.

        // This is somewhat a hack. See the assignment operator for
        // SubTableType in core/table/sub_table.h
        SubTableType safe_free = circular_buffer.front();

        // Push back the freed-up cache block retrieved from the
        // subtable that is about to be evicted.
        free_cache_blocks_[process_id].push_back(safe_free.cache_block_id());

        // Evict the subtable.
        circular_buffer.pop_front();
      }

      // Pop the list of free cache block IDs.
      free_cache_block_id = free_cache_blocks_[process_id].back();
      free_cache_blocks_[process_id].pop_back();

      // It is important that the number of attributes is multiplied
      // here.
      int mapping_offset = free_cache_block_id * cache_block_size_;
      int point_offset = mapping_offset * local_table_->n_attributes();
      point_cache_block_alias->Alias(
        point_cache_[process_id].ptr() + point_offset,
        local_table_->n_attributes(), cache_block_size_);
      *old_from_new_cache_alias = old_from_new_cache_[process_id] +
                                  mapping_offset;

      return free_cache_block_id;
    }

    /** @brief Pushes a new subtable to the given circular buffer,
     *         evicting a pre-existing subtable and destroying it
     *         appropriately if necessary.
     */
    void push_back_(
      int process_id, SubTableType &sub_table_in) {

      boost::circular_buffer<SubTableType> &circular_buffer =
        received_subtables_[ process_id ];

      // Check if the buffer is full. If so, we need to take out the
      // head element (which is going to be overwritten) and destruct
      // it manually since Boost circular buffer does not do so
      // automatically.
      if(circular_buffer.full()) {

        // This is somewhat a hack. See the assignment operator for
        // SubTableType in core/table/sub_table.h
        SubTableType safe_free = circular_buffer.front();

        // Push back the freed-up cache block retrieved from the
        // subtable that is about to be evicted.
        free_cache_blocks_[process_id].push_back(safe_free.cache_block_id());
      }
      circular_buffer.push_back(sub_table_in);
    }

    /** @brief Prints the existing subtables in the cache.
     */
    void PrintSubTables_(boost::mpi::communicator &world) {
      printf("\n\nProcess %d owns the subtables:\n", world.rank());
      for(unsigned int i = 0; i < received_subtables_.size(); i++) {
        for(int j = 0; j < received_subtables_[i].size(); j++) {
          printf(
            "%d %d %d\n", i,
            received_subtables_[i][j].table()->get_tree()->begin(),
            received_subtables_[i][j].table()->get_tree()->count());
        }
      }
    }

  public:

    /** @brief The destructor.
     */
    ~TableExchange() {
      for(unsigned int i = 0; i < point_cache_.size(); i++) {
        if(old_from_new_cache_[i] != NULL) {
          if(core::table::global_m_file_) {
            core::table::global_m_file_->DestroyPtr(old_from_new_cache_[i]);
          }
          else {
            delete[](old_from_new_cache_[i]);
          }
        }
      }
    }

    /** @brief Finds a table with given MPI rank and the beginning and
     *         the count.
     */
    SubTableType *FindSubTable(int process_id, int begin, int count) {
      for(typename boost::circular_buffer<SubTableType>::iterator
          it = received_subtables_[process_id].begin();
          it != received_subtables_[process_id].end(); it++) {
        if(
          it->table()->get_tree()->begin() == begin &&
          it->table()->get_tree()->count() == count) {

          // Put the found subtable to the end.
          SubTableType subtable_copy = *it;
          (*it) = received_subtables_[process_id].front();
          received_subtables_[process_id].pop_front();
          received_subtables_[process_id].push_back(subtable_copy);
          return &(received_subtables_[process_id].back());
        }
      }

      // Returns a NULL pointer for the non-existing table request.
      return NULL;
    }

    /** @brief Initialize the all-to-all exchange object with a
     *         distributed table and the cache size.
     */
    void Init(
      boost::mpi::communicator &world,
      TableType &local_table_in,
      int leaf_size_in,
      int max_num_levels_to_serialize_in,
      int max_num_work_to_dequeue_per_stage_in) {

      // Set the local table.
      local_table_ = &local_table_in;

      // Compute the size of each cache block, which is a function of
      // the leaf nodes owned by a subtree times each leaf node size.
      cache_block_size_ = (1 << max_num_levels_to_serialize_in) * leaf_size_in;

      // Compute the number of cache blocks allocated per process. The
      // rule is that each process gets at least twice the number of
      // work that is dequeued per stage so that there is some
      // progress in the computation, but we try to limit the number
      // of cache blocks across all processes to be 100.
      num_cache_blocks_per_process_ =
        std::max(
          2 * max_num_work_to_dequeue_per_stage_in, 100 / world.size());

      // Compute the total cache size.
      total_cache_size_per_process_ = num_cache_blocks_per_process_ *
                                      cache_block_size_;

      if(world.rank() == 0) {
        printf(
          "Cache block size (the max number of points serialized per "
          "sub-tree): %d\n", cache_block_size_);
        printf(
          "Number of cache blocks per process: %d\n",
          num_cache_blocks_per_process_);
        printf(
          "Total cache size per process: %d\n", total_cache_size_per_process_);
      }

      // Preallocate the point cache.
      point_cache_.resize(world.size());
      old_from_new_cache_.resize(world.size());
      received_subtables_.resize(world.size());
      for(int i = 0; i < world.size(); i++) {
        received_subtables_[i].set_capacity(num_cache_blocks_per_process_);
      }

      // Allocate the free cache block list.
      free_cache_blocks_.resize(world.size());
      for(int i = 0; i < world.size(); i++) {
        if(i != world.rank()) {
          for(int j = 0; j < num_cache_blocks_per_process_; j++) {
            free_cache_blocks_[i].push_back(j);
          }
        }
      }

      for(int i = 0; i < world.size(); i++) {
        if(i != world.rank()) {
          point_cache_[i].Init(
            local_table_->n_attributes(),
            total_cache_size_per_process_);
          old_from_new_cache_[i] =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            typename TableType::OldFromNewIndexType > (
              total_cache_size_per_process_) :
            new OldFromNewIndexType[ total_cache_size_per_process_ ];
        }
        else {
          old_from_new_cache_[i] = NULL;
        }
      }
    }

    /** @brief The all-to-all exchange of subtables among all MPI
     *         processes. Up to a given number of levels of subtrees
     *         are exchanged per request.
     */
    bool AllToAll(
      boost::mpi::communicator &world,
      int max_num_levels_to_serialize,
      std::vector <
      std::vector< std::pair<int, int> > > &receive_requests) {

      // The gathered request lists to send to each process.
      std::vector< std::vector< std::pair<int, int> > > send_requests;

      // Do an all-reduce to check whether you are done.
      bool all_done_flag = true;
      bool local_done_flag = true;
      for(unsigned int i = 0;
          local_done_flag && i < receive_requests.size(); i++) {
        local_done_flag = (receive_requests[i].size() == 0);
      }
      boost::mpi::all_reduce(
        world, local_done_flag, all_done_flag, std::logical_and<bool>());
      if(all_done_flag) {
        return true;
      }

      // Each process gathers the list of requests: (node
      // begin, node count) pairs.
      boost::mpi::all_to_all(
        world, receive_requests, send_requests);

      // Prepare the list of subtables to send.
      std::vector< SubTableListType > send_subtables(
        send_requests.size());
      for(unsigned int j = 0; j < send_requests.size(); j++) {
        for(unsigned int i = 0; i < send_requests[j].size(); i++) {
          int begin = send_requests[j][i].first;
          int count = send_requests[j][i].second;
          send_subtables[j].push_back(
            local_table_,
            local_table_->get_tree()->FindByBeginCount(begin, count),
            max_num_levels_to_serialize, false);
        }
      }

      // Prepare the subtable list to be received. Right now, we
      // just receive one subtable per process.
      std::vector< SubTableListType > received_subtables_in_this_round;
      received_subtables_in_this_round.resize(world.size());
      for(unsigned int j = 0; j < receive_requests.size(); j++) {
        for(unsigned int i = 0; i < receive_requests[j].size(); i++) {

          // Get a free cache block.
          core::table::DenseMatrix point_cache_alias;
          OldFromNewIndexType *old_from_new_cache_alias = NULL;
          int free_cache_block_id =
            this->get_free_cache_block_(
              j, &point_cache_alias, &old_from_new_cache_alias);

          // Allocate the cache block to the subtable that is about
          // to be received.
          received_subtables_in_this_round[j].push_back(
            j, point_cache_alias, old_from_new_cache_alias,
            free_cache_block_id, cache_block_size_,
            max_num_levels_to_serialize, false);
        }
      }

      // All-to-all to exchange the subtables.
      boost::mpi::all_to_all(
        world, send_subtables, received_subtables_in_this_round);

      // Add the new subtables to the existing cache.
      for(int j = 0; j < world.size(); j++) {
        for(unsigned int i = 0;
            i < received_subtables_in_this_round[j].size(); i++) {

          // Put the fixed subtables into the list.
          this->push_back_(
            j, received_subtables_in_this_round[j][i]);
        }
      }

      return false;
    }
};
}
}

#endif
