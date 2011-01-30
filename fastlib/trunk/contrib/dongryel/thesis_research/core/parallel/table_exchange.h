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

    std::vector< core::table::DenseMatrix > point_cache_;

    std::vector< typename TableType::OldFromNewIndexType *> old_from_new_cache_;

    std::vector<int *> new_from_old_cache_;

    /** @brief The circular buffer that acts as the cache of received
     *         subtables.
     */
    std::vector< boost::circular_buffer<SubTableType> > received_subtables_;

  private:

    /** @brief Pushes a new subtable to the given circular buffer,
     *         evicting a pre-existing subtable and destroying it
     *         appropriately if necessary.
     */
    void push_back_(
      boost::circular_buffer<SubTableType> &circular_buffer,
      SubTableType &sub_table_in) {

      // Check if the buffer is full. If so, we need to take out the
      // head element (which is going to be overwritten) and destruct
      // it manually since Boost circular buffer does not do so
      // automatically.
      if(circular_buffer.full()) {

        // This is somewhat a hack. See the assignment operator for
        // SubTableType in core/table/sub_table.h
        SubTableType safe_free = circular_buffer.front();
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

    /** @brief Extracts one request from a set of requests.
     */
    void ExtractUnitRequests_(
      std::vector< std::vector< std::pair<int, int> > > &requests,
      std::vector< std::vector< std::pair<int, int> > > *sub_requests) {

      for(unsigned int i = 0; i < requests.size(); i++) {
        if(requests[i].size() > 0) {
          if(sub_requests != NULL) {
            (*sub_requests)[i].push_back(requests[i].back());
          }
          requests[i].pop_back();
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
            core::table::global_m_file_->DestroyPtr(new_from_old_cache_[i]);
          }
          else {
            delete[](old_from_new_cache_[i]);
            delete[](new_from_old_cache_[i]);
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
      const DistributedTableType &distributed_table,
      int subtable_cache_size_per_process) {

      // Preallocate the point cache.
      point_cache_.resize(world.size());
      old_from_new_cache_.resize(world.size());
      new_from_old_cache_.resize(world.size());
      received_subtables_.resize(world.size());
      for(int i = 0; i < world.size(); i++) {
        received_subtables_[i].set_capacity(subtable_cache_size_per_process);
      }

      for(int i = 0; i < world.size(); i++) {
        if(i != world.rank()) {
          point_cache_[i].Init(
            distributed_table.n_attributes(),
            distributed_table.local_n_entries(i));
          old_from_new_cache_[i] =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            typename TableType::OldFromNewIndexType > (
              distributed_table.local_n_entries(i)) :
            new OldFromNewIndexType[ distributed_table.local_n_entries(i)];
          new_from_old_cache_[i] =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            int > (
              distributed_table.local_n_entries(i)) :
            new int[ distributed_table.local_n_entries(i)];
        }
        else {
          old_from_new_cache_[i] = NULL;
          new_from_old_cache_[i] = NULL;
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
      TableType &local_table,
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

      // The main while loop to exchange subtables.
      while(true) {

        // Dequeue a subtable list for each process.
        std::vector< std::vector< std::pair<int, int> > > sub_send_requests(
          send_requests.size());
        ExtractUnitRequests_(send_requests, &sub_send_requests);

        // Prepare the list of subtables, and do another all_to_all.
        std::vector< SubTableListType > send_subtables(
          sub_send_requests.size());
        for(unsigned int j = 0; j < sub_send_requests.size(); j++) {
          for(unsigned int i = 0; i < sub_send_requests[j].size(); i++) {
            int begin = sub_send_requests[j][i].first;
            int count = sub_send_requests[j][i].second;
            send_subtables[j].push_back(
              &local_table,
              local_table.get_tree()->FindByBeginCount(begin, count),
              max_num_levels_to_serialize);
          }
        }

        // Prepare the subtable list to be received. Right now, we
        // just receive one subtable per process.
        std::vector< SubTableListType > received_subtables_in_this_round;
        received_subtables_in_this_round.resize(world.size());
        for(unsigned int j = 0; j < receive_requests.size(); j++) {
          if(receive_requests[j].size() > 0) {
            received_subtables_in_this_round[j].push_back(
              j, point_cache_[j], old_from_new_cache_[j],
              new_from_old_cache_[j], max_num_levels_to_serialize);
          }
        }

        // All-to-all to exchange the subtables.
        boost::mpi::all_to_all(
          world, send_subtables, received_subtables_in_this_round);

        // After receiving, each item in the receive request is
        // popped.
        ExtractUnitRequests_(
          receive_requests,
          (std::vector< std::vector< std::pair<int, int> > > *) NULL);

        // Add the new subtables to the existing cache.
        for(int j = 0; j < world.size(); j++) {
          for(unsigned int i = 0;
              i < received_subtables_in_this_round[j].size(); i++) {

            // Put the fixed subtables into the list.
            this->push_back_(
              received_subtables_[j], received_subtables_in_this_round[j][i]);
          }
        }

        // If the received/sent tables are none, then quit.
        bool nothing_exchanged = true;
        bool global_nothing_exchanged = true;
        for(int i = 0; nothing_exchanged && i < world.size(); i++) {
          nothing_exchanged =
            (received_subtables_in_this_round[i].size() == 0 &&
             send_subtables[i].size() == 0);
        }
        boost::mpi::all_reduce(
          world, nothing_exchanged,
          global_nothing_exchanged, std::logical_and<bool>());
        if(global_nothing_exchanged) {
          break;
        }
      } // end of the loop.

      return false;
    }
};
}
}

#endif
