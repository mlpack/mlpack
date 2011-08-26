/** @file distributed_dualtree_task_list.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H

#include <boost/serialization/serialization.hpp>
#include <map>
#include <vector>
#include "core/parallel/distributed_dualtree_task.h"
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template < typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename QueryResultType >
class DistributedDualtreeTaskQueue;

template<typename TaskPriorityQueueType, typename QueryResultType>
class DistributedDualtreeTaskList {

  public:

    typedef typename TaskPriorityQueueType::value_type TaskType;

    typedef typename TaskType::TableType TableType;

    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::TreeType TreeType;

    typedef typename SubTableType::SubTableIDType ValueType;

    typedef ValueType KeyType;

    struct ComparatorType {
      bool operator()(const KeyType &k1, const KeyType &k2) const {
        if(k1.get<0>() < k2.get<0>()) {
          return -1;
        }
        else if(k1.get<0>() > k2.get<0>()) {
          return 1;
        }
        else {
          if(k1.get<1>() < k2.get<1>()) {
            return -1;
          }
          else if(k1.get<1>() > k2.get<1>()) {
            return 1;
          }
          else {
            if(k1.get<2>() < k2.get<2>()) {
              return -1;
            }
            else if(k1.get<2>() > k2.get<2>()) {
              return 1;
            }
            else {
              return 0;
            }
          }
        }
      }
    };

    typedef std::map< KeyType, ValueType, ComparatorType > MapType;

  private:

    int destination_rank_;

    MapType id_to_position_map_;

    unsigned long int remaining_extra_points_to_hold_;

    std::vector<SubTableType> sub_tables_;

  private:

    bool FindSubTable_(const KeyType &subtable_id) {
      return
        id_to_position_map_.find(subtable_id) != id_to_position_map_.end();
    }

    /** @brief Removes the subtable with the given ID.
     */
    void pop_(const KeyType &subtable_id) {

      // Find the position in the subtable list.
      ValueType last_subtable_id = sub_tables_.back().subtable_id();
      typename MapType::iterator remove_position_it =
        this->id_to_position_map_.find(subtable_id);
      int remove_position = -1;
      if(remove_position_it != id_to_position_map_.end()) {
        remove_position = *remove_position_it;
      }

      if(remove_position >= 0) {

        // Overwrite with the last subtable in the list and decrement.
        remaining_extra_points_to_hold_ +=
          sub_tables_[remove_position].start_node()->count();
        id_to_position_map_.erase(sub_tables_[remove_position].subtable_id());
        sub_tables_[ remove_position ].Alias(sub_tables_.back());
        sub_tables_.pop_back();
        if(sub_tables_.size() > 0) {
          id_to_position_map_[ last_subtable_id ] = remove_position;
        }
        else {
          id_to_position_map_.erase(last_subtable_id);
        }
      }
    }

    /** @brief Returns true if the subtable can be transferred within
     *         the limit.
     */
    bool push_back_(SubTableType &test_subtable_in) {

      // If already pushed, then return.
      KeyType subtable_id = test_subtable_in.subtable_id();
      if(FindSubTable_(subtable_id)) {
        return true;
      }

      // Otherwise, try to see whether it can be stored.
      else if(test_subtable_in.start_node()->count() <= remaining_extra_points_to_hold_) {
        sub_tables_.resize(sub_tables_.size() + 1);
        sub_tables_.back().Alias(test_subtable_in);
        id_to_position_map_[subtable_id] = sub_tables_.size() - 1;
        remaining_extra_points_to_hold_ -=
          test_subtable_in.start_node()->count();
        return true;
      }
      return false;
    }

  public:

    DistributedDualtreeTaskList() {
      destination_rank_ = 0;
      remaining_extra_points_to_hold_ = 0;
    }

    void Init(
      int destination_rank_in,
      unsigned long int remaining_extra_points_to_hold_in) {
      destination_rank_ = destination_rank_in;
      remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;
    }

    /** @brief Returns true if the entire task list was grabbed for
     *         the given query subtree.
     */
    template<typename DistributedTableType>
    bool push_back(
      boost::mpi::communicator &world,
      core::parallel::DistributedDualtreeTaskQueue <
      DistributedTableType,
      TaskPriorityQueueType, QueryResultType > &distributed_task_queue_in,
      int probe_index) {

      bool empty_flag = true;
      int num_reference_subtables = 0;

      // First, we need to serialize the query subtree.
      SubTableType &query_subtable =
        distributed_task_queue_in.query_subtable(probe_index);
      if(! this->push_back_(query_subtable)) {
        empty_flag = false;
      }

      // And its associated reference sets.
      while(empty_flag && distributed_task_queue_in.size(probe_index) > 0) {
        const TaskType &test_task =
          distributed_task_queue_in.top(probe_index);
        if(this->push_back_(test_task.reference_subtable())) {
          empty_flag = false;
        }
        else {

          // Pop from the list and we need to release it from the
          // cache.
          distributed_task_queue_in.ReleaseCache(
            world, test_task.reference_subtable().cache_block_id(), 1);
          distributed_task_queue_in.pop(probe_index);
          num_reference_subtables++;
        }
      }

      // If no reference subtable was pushed in, there is no point in
      // sending the query subtable.
      if(num_reference_subtables == 0) {
        this->pop_(query_subtable.subtable_id());
      }
      else {

        // Otherwise, lock the query subtable.
        distributed_task_queue_in.LockQuerySubtree(
          probe_index, destination_rank_);
      }
      return empty_flag;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the number of subtables transferred.
      int num_subtables = sub_tables_.size();
      ar & num_subtables;
      if(num_subtables > 0) {
        for(int i = 0; i < num_subtables; i++) {
          ar & sub_tables_[i];
        }
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of subtables transferred.
      int num_subtables;
      ar & num_subtables;

      if(num_subtables > 0) {
        sub_tables_.resize(num_subtables);
        for(int i = 0; i < num_subtables; i++) {
          ar & sub_tables_[i];
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif
