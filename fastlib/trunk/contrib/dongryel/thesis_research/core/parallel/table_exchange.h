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
template<typename DistributedTableType, typename SubTableListType>
class TableExchange {
  public:

    typedef typename DistributedTableType::TableType TableType;

    typedef typename SubTableListType::SubTableType SubTableType;

    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

  private:
    std::vector< core::table::DenseMatrix > point_cache_;

    std::vector< typename TableType::OldFromNewIndexType *> old_from_new_cache_;

    std::vector<int *> new_from_old_cache_;

    std::vector< boost::circular_buffer<SubTableType> > received_subtables_;

  public:

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

    SubTableType *FindSubTable(int process_id, int begin, int count) {

      // Naive search, but probably should use a STL map here...
      for(int i = static_cast<int>(received_subtables_[process_id].size()) - 1;
          i >= 0; i--) {
        if(
          received_subtables_[
            process_id][i].table()->get_tree()->begin() == begin &&
          received_subtables_[
            process_id][i].table()->get_tree()->count() == count) {
          return &(received_subtables_[process_id][i]);
        }
      }

      // Returns a NULL pointer for the non-existing table request.
      return NULL;
    }

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

    bool AllToAll(
      boost::mpi::communicator &world,
      int max_num_levels_to_serialize,
      TableType &local_table,
      const std::vector <
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

      // Prepare the list of subtables, and do another all_to_all.
      std::vector< SubTableListType > send_subtables;
      send_subtables.resize(send_requests.size());
      for(unsigned int j = 0; j < send_requests.size(); j++) {
        for(unsigned int i = 0; i < send_requests[j].size(); i++) {
          int begin = send_requests[j][i].first;
          int count = send_requests[j][i].second;
          send_subtables[j].push_back(
            &local_table,
            local_table.get_tree()->FindByBeginCount(begin, count),
            max_num_levels_to_serialize);
        }
      }

      // Push in the received subtables.
      std::vector< SubTableListType > received_subtables_in_this_round;
      received_subtables_in_this_round.resize(world.size());
      for(unsigned int j = 0; j < receive_requests.size(); j++) {
        for(unsigned int i = 0; i < receive_requests[j].size(); i++) {
          received_subtables_in_this_round[j].push_back(
            j, point_cache_[j], old_from_new_cache_[j], new_from_old_cache_[j],
            max_num_levels_to_serialize);
        }
      }
      boost::mpi::all_to_all(
        world, send_subtables, received_subtables_in_this_round);

      // Combine.
      for(int j = 0; j < world.size(); j++) {
        for(unsigned int i = 0; i < received_subtables_in_this_round[j].size();
            i++) {
          received_subtables_[j].push_back(
            received_subtables_in_this_round[j][i]);
        }
      }
      return false;
    }
};
}
}

#endif
