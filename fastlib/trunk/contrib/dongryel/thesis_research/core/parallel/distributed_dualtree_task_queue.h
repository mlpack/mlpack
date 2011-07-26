/** @file distributed_dualtree_task_queue.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_QUEUE_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_QUEUE_H

#include <deque>
#include <vector>
#include "core/math/range.h"
#include "core/parallel/table_exchange.h"

namespace core {
namespace parallel {

template<typename DistributedTableType, typename TaskPriorityQueueType>
class DistributedDualtreeTaskQueue {
  public:

    typedef typename DistributedTableType::TableType TableType;

    typedef typename DistributedTableType::TreeType TreeType;

    typedef core::parallel::TableExchange<DistributedTableType> TableExchangeType;

  private:

    std::vector< TreeType *> local_query_subtrees_;

    std::deque<bool> local_query_subtree_locks_;

    std::vector< TaskPriorityQueueType > tasks_;

    std::deque< bool > split_subtree_after_unlocking_;

    TableExchangeType *table_exchange_;

    int num_remaining_tasks_;

  private:

    template<typename MetricType>
    void split_subtree_(const MetricType &metric_in, int subtree_index) {

      // After splitting, the current index will have the left child
      // and the right child will be appended to the end of the list
      // of trees, plus duplicating the reference tasks along the way.
      TreeType *left = local_query_subtrees_[subtree_index]->left();
      TreeType *right = local_query_subtrees_[subtree_index]->right();

      local_query_subtrees_[subtree_index] = left;

      // Grow the list of local query subtrees.
      local_query_subtrees_.push_back(right);
      local_query_subtree_locks_.push_back(false);
      split_subtree_after_unlocking_.push_back(false);

      // Adjust the list of tasks.
      std::vector<TaskType> prev_tasks;
      while(tasks_[subtree_index].size() > 0) {
        std::pair<TaskType, int> task_pair;
        this->DequeueTask(subtree_index, &task_pair, false);
        prev_tasks.push_back(task_pair.first);
      }
      tasks_.resize(tasks_.size() + 1);
      for(unsigned int i = 0; i < prev_tasks.size(); i++) {
        boost::tuple<TableType *, TreeType *, int> reference_table_node_pair(
          prev_tasks[i].reference_table(),
          prev_tasks[i].reference_start_node(), prev_tasks[i].cache_id());
        this->PushTask(metric_in, subtree_index, reference_table_node_pair);
        this->PushTask(
          metric_in, local_query_subtrees_.size() - 1,
          reference_table_node_pair);

        // After splitting, it is important to lock each of the
        // reference frontier of the original tree one more time.
        table_exchange_->LockCache(prev_tasks[i].cache_id(), 1);
      }
    }

  public:

    typedef typename TaskPriorityQueueType::value_type TaskType;

  public:

    DistributedDualtreeTaskQueue() {
      num_remaining_tasks_ = 0;
      table_exchange_ = NULL;
    }

    bool is_empty() const {
      printf("Number of remaining tasks: %d\n", num_remaining_tasks_);
      return num_remaining_tasks_ == 0;
    }

    int size() const {
      return local_query_subtrees_.size();
    }

    template<typename MetricType>
    void UnlockQuerySubtree(const MetricType &metric_in, int subtree_index) {

      // Unlock the query subtree.
      local_query_subtree_locks_[ subtree_index ] = false;

      // If the splitting was requested and the tree is a non-leaf,
      // then split the query subtree.
      if(split_subtree_after_unlocking_[ subtree_index ] &&
          (! local_query_subtrees_[subtree_index]->is_leaf())) {
        if(tasks_[subtree_index].size() > 0) {
          split_subtree_(metric_in, subtree_index);
        }
        split_subtree_after_unlocking_[ subtree_index ] = false;
      }
    }

    void Init(
      TableType *local_query_table, int max_query_subtree_size,
      TableExchangeType &table_exchange_in) {

      // For each process, break up the local query tree into a list of
      // subtree query lists.
      local_query_table->get_frontier_nodes(
        max_query_subtree_size, &local_query_subtrees_);

      // Initialize the other member variables.
      local_query_subtree_locks_.resize(local_query_subtrees_.size());
      tasks_.resize(local_query_subtrees_.size());
      split_subtree_after_unlocking_.resize(local_query_subtrees_.size());
      for(unsigned int i = 0; i < local_query_subtrees_.size(); i++) {
        local_query_subtree_locks_[i] = false;
        split_subtree_after_unlocking_[i] = false;
      }

      // Pointer to the exchange mechanism.
      table_exchange_ = &table_exchange_in;
    }

    template<typename MetricType>
    void PushTask(
      const MetricType &metric_in,
      int push_index,
      boost::tuple<TableType *, TreeType *, int> &reference_table_node_pair) {

      // Compute the priority and push in.
      core::math::Range squared_distance_range(
        local_query_subtrees_[push_index]->bound().RangeDistanceSq(
          metric_in, reference_table_node_pair.get<1>()->bound()));
      TaskType new_task(
        local_query_subtrees_[push_index],
        reference_table_node_pair.get<0>(),
        reference_table_node_pair.get<1>(),
        reference_table_node_pair.get<2>(),
        - squared_distance_range.mid());
      tasks_[ push_index].push(new_task);

      // Increment the number of tasks.
      num_remaining_tasks_++;
    }

    void DequeueTask(
      int probe_index, std::pair<TaskType, int> *task_out,
      bool lock_query_subtree_in) {

      // Try to dequeue a task from the given query subtree if it is
      // not locked yet. Otherwise, request it to be split in the next
      // iteration.
      if(tasks_[probe_index].size() > 0) {
        if(! local_query_subtree_locks_[ probe_index ]) {

          // Copy the task and the query subtree number.
          task_out->first = tasks_[ probe_index ].top();
          task_out->second = probe_index;

          // Pop the task from the priority queue after copying and
          // put a lock on the query subtree.
          tasks_[ probe_index ].pop();
          local_query_subtree_locks_[ probe_index ] = lock_query_subtree_in;

          // Decrement the number of tasks.
          num_remaining_tasks_--;
        }
        else {

          // Otherwise request the query subtree to be split after it
          // is unlocked so that this thread has more chances for
          // giving a work in the next iteration.
          split_subtree_after_unlocking_[probe_index] = true;
        }
      }
    }
};
}
}

#endif
