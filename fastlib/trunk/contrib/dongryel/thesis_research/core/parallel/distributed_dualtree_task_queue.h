/** @file distributed_dualtree_task_queue.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_QUEUE_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_QUEUE_H

#include <boost/shared_ptr.hpp>
#include <deque>
#include <omp.h>
#include <vector>
#include "core/math/range.h"
#include "core/parallel/disjoint_int_intervals.h"
#include "core/parallel/distributed_dualtree_task_list.h"
#include "core/parallel/dualtree_load_balance_request.h"
#include "core/parallel/scoped_omp_lock.h"
#include "core/parallel/table_exchange.h"

namespace core {
namespace parallel {

template < typename DistributedTableType,
         typename TaskPriorityQueueType,
         typename QueryResultType >
class DistributedDualtreeTaskQueue {
  public:

    /** @brief The table type used in the exchange process.
     */
    typedef typename DistributedTableType::TableType TableType;

    /** @brief The iterator type.
     */
    typedef typename TableType::TreeIterator TreeIteratorType;

    /** @brief The tree type used in the exchange process.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The subtable type used in the exchange process.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The ID of subtables.
     */
    typedef typename SubTableType::SubTableIDType SubTableIDType;

    /** @brief The routing request type.
     */
    typedef core::parallel::RouteRequest<SubTableType> SubTableRouteRequestType;

    /** @brief The table exchange type.
     */
    typedef core::parallel::TableExchange <
    DistributedTableType, TaskPriorityQueueType, QueryResultType > TableExchangeType;

    /** @brief The type of the distributed task queue.
     */
    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType,
    TaskPriorityQueueType,
    QueryResultType > DistributedDualtreeTaskQueueType;

    typedef typename TaskPriorityQueueType::value_type TaskType;

    typedef core::parallel::DistributedDualtreeTaskList <
    DistributedTableType,
    TaskPriorityQueueType,
    QueryResultType > TaskListType;

  private:

    /** @brief Used for prioritizing tasks.
     */
    static const int process_rank_favor_factor_ = 0;

  private:

    /** @brief Assigned work for each query subtable.
     */
    std::vector <
    boost::shared_ptr<core::parallel::DisjointIntIntervals> > assigned_work_;

    /** @brief The number of remaining tasks on the current MPI
     *         process.
     */
    int num_remaining_tasks_;

    /** @brief The maximum number of working threads on the current
     *         MPI process.
     */
    int num_threads_;

    /** @brief The rank of the MPI process from which every query
     *         subtable/query result is derived. If not equal to the
     *         current MPI process rank, these must be written back
     *         when the task queue runs out.
     */
    std::vector<int> originating_ranks_;

    /** @brief The query result objects that correspond to each query
     *         subtable.
     */
    std::vector< boost::shared_ptr<QueryResultType> > query_results_;

    /** @brief The query subtable corresponding to the disjoint set of
     *         work to do for the current MPI process.
     */
    std::vector< boost::shared_ptr<SubTableType> > query_subtables_;

    /** @brief The rank of the MPI process that has a lock on the
     *         given query subtable.
     */
    std::deque<int> query_subtree_locks_;

    /** @brief The remaining global work for each query subtable.
     */
    std::vector< unsigned long int > remaining_work_for_query_subtables_;

    /** @brief The mechanism for exchanging data among all MPI
     *         processes.
     */
    TableExchangeType table_exchange_;

    /** @brief The task queue for each query subtable.
     */
    std::vector< boost::shared_ptr<TaskPriorityQueueType> > tasks_;

    /** @brief The lock that must be acquired among the threads on the
     *         same MPI process to access the queue.
     */
    omp_nest_lock_t task_queue_lock_;

    /** @brief The remaining global computation being kept track on
     *         this MPI process. If this reaches zero, then this
     *         process can exit the computation.
     */
    unsigned long int remaining_global_computation_;

    /** @brief The remaining local computation on this MPI
     *         process. Used for dynamic load balancing.
     */
    unsigned long int remaining_local_computation_;

    /** @brief If the remaining local computation falls below this
     *         level, then the MPI process tries to steal some work
     *         from its neighbor.
     */
    unsigned long int load_balancing_trigger_level_;

  private:

    void GrowSlots_() {
      assigned_work_.resize(assigned_work_.size() + 1);
      originating_ranks_.resize(originating_ranks_.size() + 1);
      query_results_.resize(query_results_.size() + 1);
      query_subtables_.resize(query_subtables_.size() + 1);
      query_subtree_locks_.resize(query_subtree_locks_.size() + 1);
      remaining_work_for_query_subtables_.resize(
        remaining_work_for_query_subtables_.size() + 1);
      tasks_.resize(tasks_.size() + 1);
    }

    /** @brief Flushes a query subtable to be written back to its
     *         origin.
     */
    void Flush_(int probe_index) {
      table_exchange_.QueueFlushRequest(
        *(query_subtables_[probe_index]),
        *(query_results_[probe_index]),
        originating_ranks_[probe_index]);
    }

    /** @brief Evicts a query subtable and its associated variables
     *         from a given slot.
     */
    void Evict_(int probe_index) {
      assigned_work_[probe_index] = assigned_work_.back();
      originating_ranks_[probe_index] = originating_ranks_.back();
      query_results_[probe_index] = query_results_.back();
      query_subtables_[probe_index] = query_subtables_.back();
      query_subtree_locks_[probe_index] = query_subtree_locks_.back();
      remaining_work_for_query_subtables_[probe_index] =
        remaining_work_for_query_subtables_.back();
      tasks_[probe_index] = tasks_.back();

      assigned_work_.pop_back();
      originating_ranks_.pop_back();
      query_results_.pop_back();
      query_subtables_.pop_back();
      query_subtree_locks_.pop_back();
      remaining_work_for_query_subtables_.pop_back();
      tasks_.pop_back();
    }

    /** @brief Tries to find more work for an additional core.
     */
    template<typename MetricType>
    void RedistributeAmongCores_(
      boost::mpi::communicator &world,
      const MetricType &metric_in) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Try to find a subtree to split.
      int split_index_query_size = 0;
      int split_index = -1;
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {
        if(query_subtree_locks_[i] < 0  &&
            (! query_subtables_[i]->start_node()->is_leaf()) &&
            tasks_[i]->size() > 0 &&
            split_index_query_size <
            query_subtables_[i]->start_node()->count())  {
          split_index_query_size = query_subtables_[i]->start_node()->count();
          split_index = i;
        }
      }
      if(split_index >= 0) {
        split_subtree_(world, metric_in, split_index);
      }
    }

    /** @brief Splits the given subtree, making an additional task
     *         queue in process.
     */
    template<typename MetricType>
    void split_subtree_(
      boost::mpi::communicator &world,
      const MetricType &metric_in, int subtree_index) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // After splitting, the current index will have the left child
      // and the right child will be appended to the end of the list
      // of trees, plus duplicating the reference tasks along the way.
      TreeType *prev_qnode = query_subtables_[subtree_index]->start_node();
      TreeType *left = prev_qnode->left();
      TreeIteratorType left_it =
        query_subtables_[subtree_index]->table()->get_node_iterator(left);
      TreeType *right = prev_qnode->right();
      TreeIteratorType right_it =
        query_subtables_[subtree_index]->table()->get_node_iterator(right);

      // Overwrite with the left child.
      query_subtables_[subtree_index]->set_start_node(left);
      query_results_[subtree_index]->Alias(left_it);

      // Grow the list of local query subtrees.
      originating_ranks_.push_back(originating_ranks_[subtree_index]);
      query_results_.push_back(
        boost::shared_ptr<QueryResultType>(new QueryResultType()));
      query_results_.back()->Alias(
        *(query_results_[subtree_index]), right_it);
      query_subtables_.push_back(
        boost::shared_ptr<SubTableType>(new SubTableType()));
      query_subtables_.back()->Alias(*(query_subtables_[subtree_index]));
      query_subtables_.back()->set_start_node(right);
      query_subtree_locks_.push_back(-1);

      // Adjust the list of tasks.
      std::vector<TaskType> prev_tasks;
      while(tasks_[subtree_index]->size() > 0) {
        std::pair<TaskType, int> task_pair;
        this->DequeueTask(world, subtree_index, &task_pair, false);
        prev_tasks.push_back(task_pair.first);
      }
      tasks_.push_back(
        boost::shared_ptr<TaskPriorityQueueType>(new TaskPriorityQueueType()));
      assigned_work_.push_back(
        boost::shared_ptr< core::parallel::DisjointIntIntervals > (
          new core::parallel::DisjointIntIntervals(
            world, *(assigned_work_[subtree_index]))));
      remaining_work_for_query_subtables_.push_back(
        remaining_work_for_query_subtables_[ subtree_index]);
      for(unsigned int i = 0; i < prev_tasks.size(); i++) {
        this->PushTask(
          world, metric_in, subtree_index,
          prev_tasks[i].reference_subtable());
        this->PushTask(
          world, metric_in, query_subtables_.size() - 1,
          prev_tasks[i].reference_subtable());

        // Lock only one time since only the query side is split.
        table_exchange_.LockCache(
          prev_tasks[i].reference_subtable_cache_block_id(), 1);
      }
    }

    /** @brief Finds the query subtable with the given index.
     */
    int FindQuerySubtreeIndex_(const SubTableIDType &query_node_id) {
      int found_index = -1;
      for(unsigned int i = 0;
          found_index < 0 && i < query_subtables_.size(); i++) {
        if(query_node_id.get<1>() ==
            query_subtables_[i]->start_node()->begin() &&
            query_node_id.get<2>() ==
            query_subtables_[i]->start_node()->count()) {
          found_index = i;
        }
      }
      return found_index;
    }

  public:

    /** @brief Initializes a new query subtable queue with its query
     *         subresult.
     */
    int PushNewQueue(
      int originating_rank_in, SubTableType &query_subtable_in,
      QueryResultType *query_subresult_in) {

      // Get more slots.
      this->GrowSlots_();
      originating_ranks_.back() = originating_rank_in;
      boost::shared_ptr< QueryResultType > tmp_query_result(
        query_subresult_in);
      query_results_.back().swap(tmp_query_result);
      query_subtables_.back()->Alias(query_subtable_in);
      query_subtree_locks_.back() = -1;
      remaining_work_for_query_subtables_.back() = 0;

      return tasks_.size() - 1;
    }

    /** @brief Pushes a given reference node onto a task list of the
     *         given query subtable.
     */
    template<typename MetricType>
    void PushTask(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      int push_index,
      SubTableType &reference_subtable) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Compute the priority and push in.
      core::math::Range squared_distance_range(
        query_subtables_[push_index]->start_node()->bound().RangeDistanceSq(
          metric_in, reference_subtable.start_node()->bound()));
      double priority = - squared_distance_range.mid() -
                        process_rank_favor_factor_ *
                        table_exchange_.process_rank(
                          world, reference_subtable.table()->rank());
      TaskType new_task(
        *(query_subtables_[push_index]),
        query_results_[ push_index ].get(),
        reference_subtable,
        priority);
      tasks_[ push_index]->push(new_task);

      // Increment the number of tasks.
      num_remaining_tasks_++;
    }

    SubTableType *FindSubTable(int cache_id) {
      return table_exchange_.FindSubTable(cache_id);
    }

    void push_subtable(
      SubTableType &subtable_in, int num_referenced_as_reference_set) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      table_exchange_.push_subtable(
        subtable_in, num_referenced_as_reference_set);
    }

    void SendLoadBalanceRequest(
      boost::mpi::communicator &world, int neighbor,
      core::parallel::DualtreeLoadBalanceRequest <
      SubTableType > **load_balance_request,
      boost::mpi::request *send_request) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      *load_balance_request =
        new core::parallel::DualtreeLoadBalanceRequest <
      SubTableType > (
        this->needs_load_balancing(),
        query_subtables_, remaining_local_computation_,
        table_exchange_.remaining_extra_points_to_hold());
      *send_request =
        world.isend(
          neighbor, core::parallel::MessageTag::LOAD_BALANCE_REQUEST,
          **load_balance_request);
    }

    /** @brief Returns whether the current MPI process needs load
     *         balancing.
     */
    bool needs_load_balancing() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return remaining_local_computation_ <= load_balancing_trigger_level_;
    }

    /** @brief Returns the query result associated with the index.
     */
    QueryResultType *query_result(int probe_index) {
      return query_results_[probe_index];
    }

    /** @brief Returns the query subtable associated with the index.
     */
    SubTableType &query_subtable(int probe_index) {
      return * (query_subtables_[probe_index]);
    }

    /** @brief The destructor.
     */
    ~DistributedDualtreeTaskQueue() {
      assigned_work_.resize(0);
      query_subtables_.resize(0);
      tasks_.resize(0);

      // Destroy the lock.
      omp_destroy_nest_lock(&task_queue_lock_);
    }

    /** @brief Returns the remaining amount of local computation.
     */
    unsigned long int remaining_local_computation() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return remaining_local_computation_;
    }

    /** @brief Returns the remaining amount of global computation.
     */
    unsigned long int remaining_global_computation() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return remaining_global_computation_;
    }

    /** @brief Decrement the remaining amount of local computation.
     */
    void decrement_remaining_local_computation(unsigned long int decrement) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      remaining_local_computation_ -= decrement;
    }

    /** @brief Decrement the remaining amount of global computation.
     */
    void decrement_remaining_global_computation(unsigned long int decrement) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      remaining_global_computation_ -= decrement;
    }

    /** @brief Releases the given cache position for the given number
     *         of times.
     */
    void ReleaseCache(
      boost::mpi::communicator &world, int cache_id, int num_times) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      table_exchange_.ReleaseCache(world, cache_id, num_times);
    }

    /** @brief Routes the data among the MPI processes, which
     *         indirectly generates tasks for the query subtables
     *         owned by the MPI process.
     */
    template<typename MetricType>
    void SendReceive(
      const MetricType &metric_in,
      boost::mpi::communicator &world,
      std::vector <
      SubTableRouteRequestType > &hashed_essential_reference_subtrees_to_send) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      table_exchange_.SendReceive(
        metric_in, world, hashed_essential_reference_subtrees_to_send);
    }

    /** @brief Generates extra tasks using the received reference
     *         subtables.
     */
    template<typename MetricType>
    void GenerateTasks(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      const std::vector <
      boost::tuple<int, int, int, int> > &received_subtable_ids) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      for(unsigned int i = 0; i < received_subtable_ids.size(); i++) {

        // Find the reference process ID and grab its subtable.
        int reference_begin = received_subtable_ids[i].get<1>();
        int reference_count = received_subtable_ids[i].get<2>();
        int cache_id = received_subtable_ids[i].get<3>();
        SubTableType *frontier_reference_subtable =
          table_exchange_.FindSubTable(cache_id);
        SubTableType alias;
        if(frontier_reference_subtable == NULL) {
          alias.Init(
            table_exchange_.local_table(),
            table_exchange_.FindByBeginCount(
              reference_begin, reference_count), false);
          alias.set_cache_block_id(cache_id);
          frontier_reference_subtable = &alias;
        }

        // For each query subtree owned by the current process, create
        // a new task if it has not already taken care of the incoming
        // reference table.
        for(int j = 0; j < this->size(); j++) {
          if(query_subtables_[j]->table()->rank() == world.rank() &&
              assigned_work_[j]->Insert(
                boost::tuple<int, int, int>(
                  frontier_reference_subtable->table()->rank(),
                  reference_begin,
                  reference_begin + reference_count))) {
            this->PushTask(
              world, metric_in, j, * frontier_reference_subtable);
            table_exchange_.LockCache(cache_id, 1);
          }
        }

      } //end of looping over each reference subtree.
    }

    /** @brief Determines whether the MPI process can terminate.
     */
    bool can_terminate() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return (remaining_global_computation_ == 0 &&
              table_exchange_.can_terminate());
    }

    /** @brief Pushes the completed computation for the given query
     *         subtable.
     */
    void push_completed_computation(
      const SubTableIDType &query_node_id,
      boost::mpi::communicator &comm,
      unsigned long int reference_count_in,
      unsigned long int quantity_in) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Subtract from the self and queue up a route message.
      remaining_global_computation_ -= quantity_in;
      remaining_local_computation_ -= quantity_in;
      table_exchange_.push_completed_computation(comm, quantity_in);

      // Update the remaining work for the query tree. Maybe the
      // searching can be sped up later.
      int found_index = this->FindQuerySubtreeIndex_(query_node_id);
      remaining_work_for_query_subtables_[found_index] -= reference_count_in;
    }

    /** @brief Pushes the completed computation for all query
     *         subtables owned by the current MPI process.
     */
    void push_completed_computation(
      boost::mpi::communicator &comm,
      unsigned long int reference_count_in,
      unsigned long int quantity_in) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Subtract from the self and queue up a route message.
      remaining_global_computation_ -= quantity_in;
      remaining_local_computation_ -= quantity_in;
      table_exchange_.push_completed_computation(comm, quantity_in);

      // Update the remaining work for all of the existing query
      // trees.
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {
        remaining_work_for_query_subtables_[i] -= reference_count_in;
      }
    }

    /** @brief Returns the remaining number of tasks on the current
     *         process.
     */
    int num_remaining_tasks() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return num_remaining_tasks_;
    }

    /** @brief Determines whether there is any remaining local
     *         computation on the current process.
     */
    bool is_empty() const {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      return (num_remaining_tasks_ == 0);
    }

    /** @brief The constructor.
     */
    DistributedDualtreeTaskQueue() {
      load_balancing_trigger_level_ = 0;
      num_remaining_tasks_ = 0;
      num_threads_ = 1;
      remaining_global_computation_ = 0;
      remaining_local_computation_ = 0;
    }

    /** @brief Returns the load balancing trigger level.
     */
    unsigned long int load_balancing_trigger_level() const {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      return load_balancing_trigger_level_;
    }

    /** @brief Returns the number of query subtables.
     */
    int size() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return query_subtables_.size();
    }

    /** @brief Locks the given query subtree for the given MPI
     *         process.
     */
    void LockQuerySubtree(int probe_index, int locking_mpi_rank) {
      query_subtree_locks_[ probe_index ] = locking_mpi_rank;
    }

    /** @brief Returns the lock to the given query subtree.
     */
    void UnlockQuerySubtree(const SubTableIDType &query_subtree_id) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      // Unlock the query subtree.
      int subtree_index = this->FindQuerySubtreeIndex_(query_subtree_id);
      query_subtree_locks_[ subtree_index ] = -1;
    }

    /** @brief Initializes the task queue.
     */
    void Init(
      boost::mpi::communicator &world,
      int max_subtree_size_in,
      DistributedTableType *query_table_in,
      DistributedTableType *reference_table_in,
      QueryResultType *local_query_result_in,
      int num_threads_in) {

      // Initialize the number of available threads.
      num_threads_ = num_threads_in;

      // Initialize the lock.
      omp_init_nest_lock(&task_queue_lock_);

      // For each process, break up the local query tree into a list of
      // subtree query lists.
      query_table_in->local_table()->get_frontier_nodes_bounded_by_number(
        4 * num_threads_in, &query_subtables_);

      // Initialize the other member variables.
      originating_ranks_.resize(query_subtables_.size());
      query_results_.resize(query_subtables_.size());
      query_subtree_locks_.resize(query_subtables_.size());
      tasks_.resize(query_subtables_.size());
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {
        originating_ranks_[i] = world.rank();
        query_results_[i] =
          boost::shared_ptr< QueryResultType >(new QueryResultType());
        TreeIteratorType qnode_it =
          query_subtables_[i]->table()->get_node_iterator(
            query_subtables_[i]->start_node());
        query_results_[i]->Alias(* local_query_result_in, qnode_it);
        query_subtree_locks_[i] = -1;
        tasks_[i] = boost::shared_ptr <
                    TaskPriorityQueueType > (new TaskPriorityQueueType());
      }

      // Initialize the table exchange.
      table_exchange_.Init(
        world, max_subtree_size_in, query_table_in, reference_table_in, this);

      // Initialize the amount of remaining computation.
      unsigned long int total_num_query_points = 0;
      unsigned long int total_num_reference_points = 0;
      for(int i = 0; i < world.size(); i++) {
        total_num_query_points += query_table_in->local_n_entries(i);
        total_num_reference_points += reference_table_in->local_n_entries(i);
      }

      // Initialize the remaining computation.
      remaining_global_computation_ =
        static_cast<unsigned long int>(total_num_query_points) *
        static_cast<unsigned long int>(total_num_reference_points);
      remaining_local_computation_ =
        static_cast<unsigned long int>(
          query_table_in->local_table()->n_entries()) *
        static_cast<unsigned long int>(total_num_reference_points);

      // Initialize the completed computation grid for each query tree
      // on this process.
      assigned_work_.resize(query_subtables_.size()) ;
      remaining_work_for_query_subtables_.resize(query_subtables_.size());
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {
        assigned_work_[i] =
          boost::shared_ptr <
          core::parallel::DisjointIntIntervals > (
            new core::parallel::DisjointIntIntervals());
        assigned_work_[i]->Init(world);
        remaining_work_for_query_subtables_[i] = total_num_reference_points;
      }

      // Load balancing trigger level is set at the half of the
      // initial local remaining work.
      load_balancing_trigger_level_ =
        remaining_local_computation_ / 2;
    }

    /** @brief Dequeues a task, optionally locking a query subtree
     *         associated with it.
     */
    template<typename MetricType>
    void DequeueTask(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      std::pair<TaskType, int> *task_out,
      bool lock_query_subtree_in) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // If the number of available task is less than the number of
      // running threads, try to get one.
      if(static_cast<int>(tasks_.size()) < num_threads_) {
        this->RedistributeAmongCores_(world, metric_in);
      }

      // Try to dequeue a task from the given query subtree if it is
      // not locked yet. Otherwise, request it to be split in the next
      // iteration.
      for(int probe_index = 0;
          probe_index < static_cast<int>(tasks_.size()); probe_index++) {

        if(this->DequeueTask(
              world, probe_index, task_out, lock_query_subtree_in)) {
          probe_index--;
        }
        if(task_out->second >= 0) {
          break;
        }
      }
    }

    /** @brief Examines the top task in the given task list.
     */
    const TaskType &top(int probe_index) const {
      return tasks_[probe_index].top();
    }

    /** @brief Removes the top task in the given task list.
     */
    void pop(int probe_index) {

      // Decrement the amount of local computation.
      remaining_local_computation_ -= tasks_[probe_index].top().work();

      // Pop.
      tasks_[probe_index].pop();

      // Decrement the number of tasks.
      num_remaining_tasks_--;
    }

    /** @brief Dequeues a task, optionally locking a query subtree
     *         associated with it.
     *
     *  @return true if the work for the query subtree in the probing
     *          index is empty.
     */
    bool DequeueTask(
      boost::mpi::communicator &world,
      int probe_index,
      std::pair<TaskType, int> *task_out,
      bool lock_query_subtree_in) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      if(tasks_[probe_index]->size() > 0) {
        if(query_subtree_locks_[ probe_index ] < 0) {

          // Copy the task and the query subtree number.
          task_out->first = tasks_[ probe_index ]->top();
          task_out->second = probe_index;

          // Pop the task from the priority queue after copying and
          // put a lock on the query subtree.
          tasks_[ probe_index ]->pop();
          query_subtree_locks_[ probe_index ] =
            lock_query_subtree_in ? world.rank() : -1 ;

          // Decrement the number of tasks.
          num_remaining_tasks_--;
        }
      }

      // Otherwise, determine whether the cleanup needs to be done.
      if(query_subtree_locks_[probe_index] < 0) {

        // If the query subtable is on the MPI process of its origin,
        if(query_subtables_[probe_index]->table()->rank() == world.rank()) {
          if(remaining_work_for_query_subtables_[probe_index] == 0) {
            this->Evict_(probe_index);
            return true;
          }
        }

        // If the query subtable is not from the MPI process of its
        // origin and it ran out of stuffs to do, flush.
        else if(tasks_[probe_index]->size() == 0) {
          this->Flush_(probe_index);
          return true;
        }
      }
      return false;
    }
};
}
}

#endif
