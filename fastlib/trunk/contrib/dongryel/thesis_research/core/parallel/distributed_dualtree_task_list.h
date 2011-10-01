/** @file distributed_dualtree_task_list.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H

#include <boost/intrusive_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <map>
#include <vector>
#include "core/parallel/distributed_dualtree_task.h"
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template < typename DistributedTableType,
         typename TaskPriorityQueueType >
class DistributedDualtreeTaskQueue;

template < typename DistributedTableType,
         typename TaskPriorityQueueType >
class DistributedDualtreeTaskList {

  public:

    typedef typename TaskPriorityQueueType::value_type TaskType;

    typedef typename TaskType::TableType TableType;

    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::TreeType TreeType;

    typedef typename SubTableType::SubTableIDType KeyType;

    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType,
    TaskPriorityQueueType > DistributedDualtreeTaskQueueType;

    typedef int ValueType;

    struct ComparatorType {
      bool operator()(const KeyType &k1, const KeyType &k2) const {
        return k1.get<0>() < k2.get<0>() ||
               (k1.get<0>() == k2.get<0>() && k1.get<1>() < k2.get<1>()) ||
               (k1.get<0>() == k2.get<0>() && k1.get<1>() == k2.get<1>() &&
                k1.get<2>() < k2.get<2>()) ;
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

    /** @brief The subtables that must be transferred. The second
     *         component tells whether the subtable is referenced as a
     *         query. The third pair denotes the number of times the
     *         subtable is referenced as a reference set.
     */
    std::vector <
    boost::tuple <
    boost::intrusive_ptr<SubTableType> , bool, int > > sub_tables_;

    /** @brief The communicator for exchanging information.
     */
    boost::mpi::communicator *world_;

  private:

    void Print_() const {

      // Print out the mapping.
      typename MapType::const_iterator it = id_to_position_map_.begin();
      printf("Mapping from subtable IDs to position:\n");
      for(; it != id_to_position_map_.end(); it++) {
        printf(
          "(%d %d %d) -> %d\n",
          it->first.get<0>(), it->first.get<1>(), it->first.get<2>(),
          it->second);
      }

      // Print out the subtables.
      for(unsigned int i = 0; i < sub_tables_.size(); i++) {
        KeyType subtable_id = sub_tables_[i].get<0>()->subtable_id();
        printf("%d: (%d %d %d)\n", i, subtable_id.get<0>(),
               subtable_id.get<1>(), subtable_id.get<2>());
      }
    }

    /** @brief Returns the position of the subtable.
     */
    bool FindSubTable_(const KeyType &subtable_id, int *position_out) {
      typename MapType::iterator it =
        id_to_position_map_.find(subtable_id);
      if(it != id_to_position_map_.end()) {
        *position_out = it->second;
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
        typename MapType::iterator remove_position_it =
          id_to_position_map_.find(subtable_id);

        // This position should be valid, if there is at least one
        // subtable.
        int remove_position = remove_position_it->second;

        // Remove as a query table.
        if(count_as_query) {
          sub_tables_[remove_position].get<1>() = false;
        }

        // Otherwise, remove as a reference table by decrementing the
        // reference count.
        else {
          sub_tables_[remove_position].get<2>()--;
        }

        // If the subtable is no longer is referenced, we have to
        // remove it.
        if((!  sub_tables_[remove_position].get<1>()) &&
            sub_tables_[remove_position].get<2>() == 0) {

          // Overwrite with the last subtable in the list and decrement.
          KeyType last_subtable_id =
            sub_tables_.back().get<0>()->subtable_id();
          id_to_position_map_.erase(
            sub_tables_[remove_position].get<0>()->subtable_id());
          sub_tables_[ remove_position ].get<0>() =
            sub_tables_.back().get<0>();
          sub_tables_[ remove_position ].get<1>() =
            sub_tables_.back().get<1>();
          sub_tables_[ remove_position ].get<2>() =
            sub_tables_.back().get<2>();
          sub_tables_.pop_back();
          id_to_position_map_.erase(last_subtable_id);
          if(remove_position != static_cast<int>(sub_tables_.size()))  {
            id_to_position_map_[ last_subtable_id ] = remove_position;
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
        if(! count_as_query) {
          sub_tables_[ existing_position ].get<2>()++;
        }
        else {
          sub_tables_[existing_position].get<0>()->Alias(test_subtable_in);
          sub_tables_[ existing_position ].get<1>() = true;
        }
        return existing_position;
      }

      // Otherwise, try to see whether it can be stored.
      else {
        sub_tables_.resize(sub_tables_.size() + 1);

        boost::intrusive_ptr< SubTableType > tmp_subtable(new SubTableType());
        sub_tables_.back().get<0>().swap(tmp_subtable);
        sub_tables_.back().get<0>()->Alias(test_subtable_in);
        if(count_as_query) {
          sub_tables_.back().get<1>() = true;
          sub_tables_.back().get<2>() = 0;
        }
        else {
          sub_tables_.back().get<1>() = false;
          sub_tables_.back().get<2>() = 1;
        }
        id_to_position_map_[subtable_id] = sub_tables_.size() - 1;
        return sub_tables_.size() - 1;
      }
    }

  public:

    int num_subtables() const {
      return sub_tables_.size();
    }

    void operator=(const DistributedDualtreeTaskList &task_list_in) {
      destination_rank_ = task_list_in.destination_rank_;
      distributed_task_queue_ = task_list_in.distributed_task_queue_;
      donated_task_list_ = task_list_in.donated_task_list_;
      id_to_position_map_ = task_list_in.id_to_position_map_;
      sub_tables_ = task_list_in.sub_tables_;
      world_ = task_list_in.world_;
    }

    DistributedDualtreeTaskList(
      const DistributedDualtreeTaskList &task_list_in) {
      this->operator=(task_list_in);
    }

    /** @brief The default constructor.
     */
    DistributedDualtreeTaskList() {
      destination_rank_ = 0;
      distributed_task_queue_ = NULL;
      world_ = NULL;
    }

    /** @brief Exports the received task list to the distributed task
     *         queue.
     */
    template<typename MetricType>
    void Export(
      boost::mpi::communicator &world,
      const MetricType &metric_in, int source_rank_in,
      DistributedDualtreeTaskQueueType *distributed_task_queue_in) {

      // Set the queue pointer.
      distributed_task_queue_ = distributed_task_queue_in;

      // Get a free slot for each subtable.
      std::vector<int> assigned_cache_indices;
      for(unsigned int i = 0; i < sub_tables_.size(); i++) {
        KeyType subtable_id =  sub_tables_[i].get<0>()->subtable_id();
        (sub_tables_[i].get<0>())->set_originating_rank(source_rank_in);
        assigned_cache_indices.push_back(
          distributed_task_queue_->push_subtable(
            *(sub_tables_[i].get<0>()), sub_tables_[i].get<2>()));
      }

      // Now push in the task list for each query subtable.
      for(unsigned int i = 0; i < donated_task_list_.size(); i++) {
        int query_subtable_position =  donated_task_list_[i].first;
        SubTableType *query_subtable_in_cache =
          distributed_task_queue_->FindSubTable(
            assigned_cache_indices[query_subtable_position]);
        int new_position =
          distributed_task_queue_->PushNewQueue(
            world, source_rank_in, *query_subtable_in_cache);

        unsigned long int num_reference_points_for_new_query_subtable = 0 ;
        for(unsigned int j = 0;
            j <  donated_task_list_[i].second.size(); j++) {
          int reference_subtable_position = donated_task_list_[i].second[j];
          SubTableType *reference_subtable_in_cache =
            distributed_task_queue_->FindSubTable(
              assigned_cache_indices[ reference_subtable_position ]);
          distributed_task_queue_->PushTask(
            world, metric_in, new_position, *reference_subtable_in_cache);
          num_reference_points_for_new_query_subtable +=
            reference_subtable_in_cache->start_node()->count();
        }

        // Set the remaining number of reference points for the new
        // query subtable.
        distributed_task_queue_->set_remaining_work_for_query_subtable(
          new_position, num_reference_points_for_new_query_subtable);
      }

      // Swap the positions so that the imported query subtables are
      // selected first.
      for(unsigned int i = 0; i < donated_task_list_.size(); i++) {
        distributed_task_queue_->assigned_work_.back().swap(
          distributed_task_queue_->assigned_work_[i]);
        distributed_task_queue_->query_subtables_.back().swap(
          distributed_task_queue_->query_subtables_[i]);
        std::swap(
          distributed_task_queue_->remaining_work_for_query_subtables_.back(),
          distributed_task_queue_->remaining_work_for_query_subtables_[i]);
        distributed_task_queue_->tasks_.back().swap(
          distributed_task_queue_->tasks_[i]);
      }
    }

    /** @brief Initializes the task list.
     */
    void Init(
      boost::mpi::communicator &world,
      int destination_rank_in,
      DistributedDualtreeTaskQueueType &distributed_task_queue_in) {
      destination_rank_ = destination_rank_in;
      distributed_task_queue_ = &distributed_task_queue_in;
      world_ = &world;
      sub_tables_.resize(0);
      id_to_position_map_.clear();
      donated_task_list_.resize(0);
    }

    /** @brief Tries to fit in as many tasks from the given query
     *         subtree.
     *
     *  @return true if the query subtable is successfully locked.
     */
    bool push_back(
      boost::mpi::communicator &world,
      int destination_rank_in, int probe_index) {

      // Return if the task queue is empty.
      if(distributed_task_queue_->size(probe_index) == 0) {
        return false;
      }

      // First, we need to serialize the query subtree.
      SubTableType &query_subtable =
        distributed_task_queue_->query_subtable(probe_index);
      int query_subtable_position;
      if((query_subtable_position =
            this->push_back_(query_subtable, true)) < 0) {
        return false;
      }
      donated_task_list_.resize(donated_task_list_.size() + 1);
      donated_task_list_.back().first = query_subtable_position;

      // And its associated reference sets.
      while(distributed_task_queue_->size(probe_index) > 0) {
        TaskType &test_task =
          const_cast<TaskType &>(distributed_task_queue_->top(probe_index));
        int reference_subtable_position;

        // If the reference subtable cannot be packed, break.
        if((reference_subtable_position =
              this->push_back_(test_task.reference_subtable(), false)) < 0) {
          break;
        }
        else {

          // Pop from the list. Releasing each reference subtable from
          // the cache is done in serialization.
          distributed_task_queue_->pop(probe_index);
          donated_task_list_.back().second.push_back(
            reference_subtable_position);
        }
      } // end of trying to empty out a query subtable list.

      // If no reference subtable was pushed in, there is no point in
      // sending the query subtable.
      if(donated_task_list_.back().second.size() == 0) {
        this->pop_(query_subtable.subtable_id(), true);
        donated_task_list_.pop_back();
        return false;
      }
      else {

        // Otherwise, lock the query subtable.
        distributed_task_queue_->LockQuerySubTable(
          probe_index, destination_rank_);
        return true;
      }
    }

    void ExportPostFreeCacheList(
      std::vector< std::pair<int, int> > *post_free_cache_list_out) {

      post_free_cache_list_out->resize(0);
      for(unsigned int i = 0; i < sub_tables_.size(); i++) {
        post_free_cache_list_out->push_back(
          std::pair<int, int> (
            sub_tables_[i].get<0>()->cache_block_id(),
            sub_tables_[i].get<2>()));
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
          ar & (*(sub_tables_[i].get<0>()));
          ar &  sub_tables_[i].get<2>();
        }

        // Save the donated task lists.
        int num_donated_lists = donated_task_list_.size();
        ar & num_donated_lists;
        for(int i = 0; i < num_donated_lists; i++) {
          int sublist_size =  donated_task_list_[i].second.size();
          ar &  donated_task_list_[i].first;
          ar & sublist_size;
          for(int j = 0; j < sublist_size; j++) {
            ar &  donated_task_list_[i].second[j];
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
          sub_tables_[i].get<0>() = new SubTableType();
          sub_tables_[i].get<0>()->Init(i, false);
          ar & (*(sub_tables_[i].get<0>()));
          ar & sub_tables_[i].get<2>();
          sub_tables_[i].get<1>() =
            (sub_tables_[i].get<0>()->query_result() != NULL);
        }

        // Load the donated task lists.
        int num_donated_lists;
        ar & num_donated_lists;
        donated_task_list_.resize(num_donated_lists);
        for(int i = 0; i < num_donated_lists; i++) {
          int sublist_size;
          ar &  donated_task_list_[i].first;
          ar & sublist_size;
          donated_task_list_[i].second.resize(sublist_size);
          for(int j = 0; j < sublist_size; j++) {
            ar &  donated_task_list_[i].second[j];
          }
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif
