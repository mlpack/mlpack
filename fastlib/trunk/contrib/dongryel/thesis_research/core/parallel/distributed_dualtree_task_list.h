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

    typedef boost::tuple<int, int, int> ValueType;

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

    MapType id_to_position_map_;

    unsigned long int remaining_extra_points_to_hold_;

    std::vector<SubTableType> sub_tables_;

  private:

    bool FindSubTable_(const boost::tuple<int, int, int> &subtable_id) {
      return
        id_to_position_map_.find(subtable_id) != id_to_position_map_.end();
    }

    /** @brief Returns true if the subtable can be transferred within
     *         the limit.
     */
    bool push_back_(SubTableType &test_subtable_in) {

      // If already pushed, then return.
      boost::tuple<int, int, int> subtable_id(
        test_subtable_in.table()->rank(),
        test_subtable_in.start_node()->begin(),
        test_subtable_in.start_node()->count());
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
      remaining_extra_points_to_hold_ = 0;
    }

    void Init(unsigned long int remaining_extra_points_to_hold_in) {
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

      // First, we need to serialize the query subtree.
      if(! this->push_back_(
            distributed_task_queue_in.query_subtable(probe_index))) {
        empty_flag = false;
      }

      /*
      // And its associated reference sets.
      while(empty_flag && query_subtree_task_queue.size() > 0) {
      	std::pair<TaskType, int> test_task;
        const TaskType &test_task = query_subtree_task_queue.top();
        if(this->push_back_(test_task.reference_subtable())) {
          empty_flag = false;
        }
        else {

          // Pop from the list and we need to release it from the
          // cache.
          table_exchange_in.ReleaseCache(
            world, test_task.reference_subtable().cache_block_id(), 1);
          table_exchange_in.decrement;
          query_subtree_task_queue.pop();
        }
      }
      */
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
