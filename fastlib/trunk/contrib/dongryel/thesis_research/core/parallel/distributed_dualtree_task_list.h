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

template < typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename QueryResultType >
class DistributedDualtreeTaskList {

  public:

    typedef typename TaskPriorityQueueType::value_type TaskType;

    typedef typename TaskType::TableType TableType;

    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::TreeType TreeType;

    typedef typename SubTableType::SubTableIDType ValueType;

    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType,
    TaskPriorityQueueType,
    QueryResultType > DistributedDualtreeTaskQueueType;

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

    /** @brief The destination MPI rank for which tasks are to be
     *         donated.
     */
    int destination_rank_;

    /** @brief The pointer to the distributed task queue.
     */
    DistributedDualtreeTaskQueueType *distributed_task_queue_;

    /** @brief The donated task list of (Q, R) pairs. Indexes the
     *         subtable positions.
     */
    std::vector< std::pair<int, std::vector<int> > > donated_task_list_;

    /** @brief The map that maps the subtable ID to the position.
     */
    MapType id_to_position_map_;

    /** @brief Denotes the number of extra points that can be
     *         transferred.
     */
    unsigned long int remaining_extra_points_to_hold_;

    /** @brief The subtables that must be transferred. The second
     *         component tells whether the subtable is referenced as a
     *         query. The third pair denotes the number of times the
     *         subtable is referenced as a reference set.
     */
    std::vector< boost::tuple<SubTableType, bool, int> > sub_tables_;

    /** @brief The communicator for exchanging information.
     */
    boost::mpi::communicator *world_;

  private:

    /** @brief Returns the position of the subtable.
     */
    bool FindSubTable_(const KeyType &subtable_id, int *position_out) {
      typename MapType::iterator it =
        id_to_position_map_.find(subtable_id);
      if(it != id_to_position_map_.end()) {
        *position_out = *it;
        return true;
      }
      else {
        *position_out = -1;
        return false;
      }
    }

    /** @brief Removes the subtable with the given ID.
     */
    void pop_(const KeyType &subtable_id, bool count_as_query) {

      if(sub_tables_.size() > 0) {

        // Find the position in the subtable list.
        ValueType last_subtable_id = sub_tables_.back().get<0>().subtable_id();
        typename MapType::iterator remove_position_it =
          this->id_to_position_map_.find(subtable_id);
        int remove_position = -1;
        if(remove_position_it != id_to_position_map_.end()) {
          remove_position = *remove_position_it;
        }

        if(remove_position >= 0) {

          if(count_as_query) {
            sub_tables_[remove_position].get<1>() = false;
          }
          else {
            // Decrement the reference count.
            sub_tables_[remove_position].get<2>()--;
          }

          if((! sub_tables_[remove_position].get<1>()) &&
              sub_tables_[remove_position].get<2>() == 0) {

            // Overwrite with the last subtable in the list and decrement.
            remaining_extra_points_to_hold_ +=
              sub_tables_[remove_position].get<0>().start_node()->count();
            id_to_position_map_.erase(
              sub_tables_[remove_position].get<0>().subtable_id());
            sub_tables_[ remove_position ].get<0>().Alias(
              sub_tables_.back().get<0>());
            sub_tables_[ remove_position ].get<1>() =
              sub_tables_.back().get<1>();
            sub_tables_[ remove_position ].get<2>() =
              sub_tables_.back().get<2>();
            sub_tables_.pop_back();
            if(sub_tables_.size() > 0) {
              id_to_position_map_[ last_subtable_id ] = remove_position;
            }
            else {
              id_to_position_map_.erase(last_subtable_id);
            }
          }
        }
      }
    }

    /** @brief Returns the assigned position of the subtable if it can
     *         be transferred within the limit.
     */
    int push_back_(SubTableType &test_subtable_in, bool count_as_query) {

      // If already pushed, then return.
      KeyType subtable_id = test_subtable_in.subtable_id();
      int existing_position;
      if(this->FindSubTable_(subtable_id, &existing_position)) {
        sub_tables_[ existing_position ].get<2>()++;
        return existing_position;
      }

      // Otherwise, try to see whether it can be stored.
      else if(test_subtable_in.start_node()->count() <=
              remaining_extra_points_to_hold_) {
        sub_tables_.resize(sub_tables_.size() + 1);
        sub_tables_.back().get<0>().Alias(test_subtable_in);
        if(count_as_query) {
          sub_tables_.back().get<1>() = true;
        }
        else {
          sub_tables_.back().get<2>() = 1;
        }
        id_to_position_map_[subtable_id] = sub_tables_.size() - 1;
        remaining_extra_points_to_hold_ -=
          test_subtable_in.start_node()->count();
        return sub_tables_.size() - 1;
      }
      return -1;
    }

  public:

    /** @brief Returns the number of extra points.
     */
    unsigned long int remaining_extra_points_to_hold() const {
      return remaining_extra_points_to_hold_;
    }

    /** @brief The default constructor.
     */
    DistributedDualtreeTaskList() {
      destination_rank_ = 0;
      remaining_extra_points_to_hold_ = 0;
      world_ = NULL;
    }

    /** @brief Initializes the task list.
     */
    void Init(
      boost::mpi::communicator &world,
      int destination_rank_in,
      unsigned long int remaining_extra_points_to_hold_in,
      core::parallel::DistributedDualtreeTaskQueue <
      DistributedTableType,
      TaskPriorityQueueType, QueryResultType > &distributed_task_queue_in) {
      destination_rank_ = destination_rank_in;
      distributed_task_queue_ = &distributed_task_queue_in;
      remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;
      world_ = &world;
    }

    /** @brief Tries to fit in as many tasks from the given query
     *         subtree.
     */
    void push_back(boost::mpi::communicator &world, int probe_index) {

      // First, we need to serialize the query subtree.
      SubTableType &query_subtable =
        distributed_task_queue_->query_subtable(probe_index);
      int query_subtable_position;
      if((query_subtable_position =
            this->push_back_(query_subtable, true)) < 0) {
        return;
      }
      donated_task_list_.resize(donated_task_list_.size() + 1);
      donated_task_list_.back().first = query_subtable_position;

      // And its associated reference sets.
      while(distributed_task_queue_->size(probe_index) > 0) {
        const TaskType &test_task = distributed_task_queue_->top(probe_index);
        int reference_subtable_position;
        if((reference_subtable_position =
              this->push_back_(test_task.reference_subtable(), false)) >= 0) {
          break;
        }
        else {

          // Pop from the list. Releasing each reference subtable from
          // the cache is done in serialization.
          distributed_task_queue_->pop(probe_index);
          donated_task_list_.back().second.push_back(
            reference_subtable_position);
        }
      }

      // If no reference subtable was pushed in, there is no point in
      // sending the query subtable.
      if(donated_task_list_.back().second.size() == 0) {
        this->pop_(query_subtable.subtable_id(), true);
        donated_task_list_.pop_back();
      }
      else {

        // Otherwise, lock the query subtable.
        distributed_task_queue_->LockQuerySubtree(
          probe_index, destination_rank_);
      }
    }

    /** @brief Saves the donated task list.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the number of subtables transferred.
      int num_subtables = sub_tables_.size();
      ar & num_subtables;
      if(num_subtables > 0) {
        for(int i = 0; i < num_subtables; i++) {
          ar & sub_tables_[i].get<0>();

          // If this is a reference subtable, then we need to release
          // it from the cache owned by the donating process.
          const_cast <
          DistributedDualtreeTaskQueueType * >(
            distributed_task_queue_)->ReleaseCache(
              *world_,
              sub_tables_[i].get<0>().cache_block_id(),
              sub_tables_[i].get<2>());
        }

        // Save the donated task lists.
        int num_donated_lists = donated_task_list_.size();
        ar & num_donated_lists;
        for(int i = 0; i < num_donated_lists; i++) {
          int sublist_size = donated_task_list_[i].second.size();
          ar & donated_task_list_[i].first;
          ar & sublist_size;
          for(int j = 0; j < sublist_size; j++) {
            ar & donated_task_list_[i].second[j];
          }
        }
      }
    }

    /** @brief Loads the donated task list.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of subtables transferred.
      int num_subtables;
      ar & num_subtables;

      if(num_subtables > 0) {
        sub_tables_.resize(num_subtables);
        for(int i = 0; i < num_subtables; i++) {

          // Need to the cache block correction later.
          sub_tables_[i].Init(i, false);
          ar & sub_tables_[i].get<0>();
          sub_tables_[i].get<1>() = true;
          sub_tables_[i].get<2>() = 1;
        }

        // Load the donated task lists.
        int num_donated_lists;
        ar & num_donated_lists;
        donated_task_list_.resize(num_donated_lists);
        for(int i = 0; i < num_donated_lists; i++) {
          int sublist_size;
          ar & donated_task_list_[i].first;
          ar & sublist_size;
          donated_task_list_[i].second.resize(sublist_size);
          for(int j = 0; j < sublist_size; j++) {
            ar & donated_task_list_[i].second[j];
          }
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif
