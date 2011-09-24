/** @file distributed_dualtree_task_queue.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_QUEUE_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_QUEUE_H

#include <boost/intrusive_ptr.hpp>
#include <deque>
#include <list>
#include <omp.h>
#include <vector>
#include "core/math/range.h"
#include "core/parallel/disjoint_int_intervals.h"
#include "core/parallel/distributed_dualtree_task_list.h"
#include "core/parallel/dualtree_load_balance_request.h"
#include "core/parallel/query_subtable_lock.h"
#include "core/parallel/scoped_omp_lock.h"
#include "core/parallel/table_exchange.h"

namespace core {
namespace parallel {

template < typename DistributedTableType,
         typename TaskPriorityQueueType >
class DistributedDualtreeTaskQueue {
  public:

    /** @brief The associated query result type.
     */
    typedef typename DistributedTableType::QueryResultType QueryResultType;

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
    DistributedTableType, TaskPriorityQueueType > TableExchangeType;

    /** @brief The type of the distributed task queue.
     */
    typedef core::parallel::DistributedDualtreeTaskQueue <
    DistributedTableType,
    TaskPriorityQueueType > DistributedDualtreeTaskQueueType;

    typedef typename TaskPriorityQueueType::value_type TaskType;

    typedef core::parallel::DistributedDualtreeTaskList <
    DistributedTableType,
    TaskPriorityQueueType > TaskListType;

    friend class QuerySubTableLock <
        DistributedTableType, TaskPriorityQueueType >;

  typedef class QuerySubTableLock <
        DistributedTableType, TaskPriorityQueueType > QuerySubTableLockType;

    typedef std::list< boost::intrusive_ptr< QuerySubTableLockType > >
    QuerySubTableLockListType;

    typedef core::parallel::DualtreeLoadBalanceRequest <
    DistributedTableType,
    TaskPriorityQueueType > DualtreeLoadBalanceRequestType;

  private:

    /** @brief Used for prioritizing tasks.
     */
    static const int process_rank_favor_factor_ = 0;

  private:

    /** @brief Assigned work for each query subtable.
     */
    std::vector <
    boost::intrusive_ptr<core::parallel::DisjointIntIntervals> > assigned_work_;

    /** @brief Whether each query subtable can be exported.
     */
    std::deque< bool > can_be_exported_;

    /** @brief The list of checked out query subtables.
     */
    QuerySubTableLockListType checked_out_query_subtables_;

    /** @brief The number of deterministic prunes.
     */
    int num_deterministic_prunes_;

    /** @brief The number of exported query subtables from this MPI
     *         process to other MPI Processes.
     */
    int num_exported_query_subtables_;

    /** @brief The number of imported query subtables from other MPI
     *         processes.
     */
    int num_imported_query_subtables_;

    /** @brief The number of probabilistic prunes.
     */
    int num_probabilistic_prunes_;

    /** @brief The number of query subtables that have received
     *         essential data from every process. These query
     *         subtables can be safely exported to other MPI processes
     *         for load balancing purposes.
     */
    int num_receive_completed_query_subtables_;

    /** @brief The number of remaining tasks on the current MPI
     *         process.
     */
    int num_remaining_tasks_;

    /** @brief The maximum number of working threads on the current
     *         MPI process.
     */
    int num_threads_;

    /** @brief The query subtable corresponding to the disjoint set of
     *         work to do for the current MPI process.
     */
    std::vector< boost::intrusive_ptr<SubTableType> > query_subtables_;

    /** @brief The remaining global computation being kept track on
     *         this MPI process. If this reaches zero, then this
     *         process can exit the computation.
     */
    unsigned long int remaining_global_computation_;

    /** @brief The remaining local computation on this MPI
     *         process. Used for dynamic load balancing.
     */
    unsigned long int remaining_local_computation_;

    /** @brief The remaining global work for each query subtable.
     */
    std::vector< unsigned long int > remaining_work_for_query_subtables_;

    /** @brief The remaining work in the priority queue currently for
     *         each query subtable.
     */
    std::vector< unsigned long int > remaining_work_in_priority_queue_;

    /** @brief The mechanism for exchanging data among all MPI
     *         processes.
     */
    TableExchangeType table_exchange_;

    /** @brief The task queue for each query subtable.
     */
    std::vector< boost::intrusive_ptr<TaskPriorityQueueType> > tasks_;

    /** @brief The lock that must be acquired among the threads on the
     *         same MPI process to access the queue.
     */
    omp_nest_lock_t task_queue_lock_;

  private:

    /** @brief Grow slots for additional query subtables.
     */
    void GrowSlots_() {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      assigned_work_.resize(assigned_work_.size() + 1);
      can_be_exported_.resize(can_be_exported_.size() + 1);
      query_subtables_.push_back(
        boost::intrusive_ptr< SubTableType > (new SubTableType()));
      remaining_work_for_query_subtables_.resize(
        remaining_work_for_query_subtables_.size() + 1);
      remaining_work_in_priority_queue_.resize(
        remaining_work_in_priority_queue_.size() + 1);
      tasks_.push_back(
        boost::intrusive_ptr <
        TaskPriorityQueueType > (new TaskPriorityQueueType()));
    }

    /** @brief Flushes a query subtable to be written back to its
     *         origin.
     */
    void Flush_(boost::mpi::communicator &world, int probe_index) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Queue and evict.
      table_exchange_.QueueFlushRequest(world, query_subtables_[probe_index]);
      num_imported_query_subtables_--;
      this->Evict_(world, probe_index);
    }

    /** @brief Evicts a query subtable and its associated variables
     *         from a given slot.
     */
    void Evict_(boost::mpi::communicator &world, int probe_index) {

      // Lock the queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      assigned_work_[probe_index] = assigned_work_.back();
      can_be_exported_[probe_index] = can_be_exported_.back();
      query_subtables_[probe_index] = query_subtables_.back();
      remaining_work_for_query_subtables_[probe_index] =
        remaining_work_for_query_subtables_.back();
      remaining_work_in_priority_queue_[ probe_index ] =
        remaining_work_in_priority_queue_.back();
      tasks_[probe_index] = tasks_.back();

      assigned_work_.pop_back();
      can_be_exported_.pop_back();
      query_subtables_.pop_back();
      remaining_work_for_query_subtables_.pop_back();
      remaining_work_in_priority_queue_.pop_back();
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
        if((! query_subtables_[i]->start_node()->is_leaf()) &&
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
      TreeType *right = prev_qnode->right();

      // Overwrite with the left child.
      query_subtables_[subtree_index]->set_start_node(left);

      // Grow the list of local query subtrees.
      query_subtables_.push_back(
        boost::intrusive_ptr<SubTableType>(new SubTableType()));
      query_subtables_.back()->Alias(*(query_subtables_[subtree_index]));
      query_subtables_.back()->set_start_node(right);

      // Adjust the list of tasks.
      std::vector<TaskType> prev_tasks;
      while(tasks_[subtree_index]->size() > 0) {
        std::pair<TaskType, int> task_pair;
        this->DequeueTask(
          world, subtree_index, &task_pair,
          (typename QuerySubTableLockListType::iterator *) NULL);
        prev_tasks.push_back(task_pair.first);
      }
      tasks_.push_back(
        boost::intrusive_ptr <
        TaskPriorityQueueType > (new TaskPriorityQueueType()));
      assigned_work_.push_back(
        boost::intrusive_ptr< core::parallel::DisjointIntIntervals > (
          new core::parallel::DisjointIntIntervals(
            world, *(assigned_work_[subtree_index]))));
      can_be_exported_.push_back(can_be_exported_[ subtree_index ]);
      remaining_work_for_query_subtables_.push_back(
        remaining_work_for_query_subtables_[ subtree_index]);
      remaining_work_in_priority_queue_.push_back(
        remaining_work_in_priority_queue_[ subtree_index ]);
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

  public:

    void set_remaining_work_for_query_subtable(
      int probe_index, unsigned long int work_in) {

      remaining_work_for_query_subtables_[ probe_index ] = work_in;
      remaining_work_in_priority_queue_[ probe_index ] = work_in;
    }

    /** @brief Synchronizes the local query subtable with the received
     *         query subtable.
     */
    void Synchronize(
      boost::mpi::communicator &world,
      SubTableType &received_query_subtable_in) {

      // Lock the queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      SubTableIDType received_query_subtable_id =
        received_query_subtable_in.subtable_id();

      // Find the checked out subtable in the list and synchronize.
      for(typename QuerySubTableLockListType::iterator it =
            checked_out_query_subtables_.begin();
          it != checked_out_query_subtables_.end(); it++) {
        if((*it)->query_subtable_->includes(received_query_subtable_in)) {
          (*it)->query_subtable_->Copy(received_query_subtable_in);

          // Now put back the synchronized part into the active queue,
          // splitting the existing checked out query subtable if
          // necessary.
          SubTableIDType comp_query_subtable_id =
            (*it)->query_subtable_->subtable_id();
          if(received_query_subtable_id.get<0>() ==
              comp_query_subtable_id.get<0>() &&
              received_query_subtable_id.get<1>() ==
              comp_query_subtable_id.get<1>() &&
              received_query_subtable_id.get<2>() ==
              comp_query_subtable_id.get<2>()) {

            assigned_work_.push_back((*it)->assigned_work_);
            can_be_exported_.push_back((*it)->can_be_exported_);
            query_subtables_.push_back((*it)->query_subtable_);
            remaining_work_for_query_subtables_.push_back(
              (*it)->remaining_work_for_query_subtable_);
            remaining_work_in_priority_queue_.push_back(
              (*it)->remaining_work_in_priority_queue_);
            tasks_.push_back((*it)->task_);
            checked_out_query_subtables_.erase(it);

            num_exported_query_subtables_--;
          }
          else {

            printf("Need to implement this case!\n");
            exit(0);
          }
          break;
        }
      }
    }

    /** @brief Returns a locked query subtable to the active pool.
     */
    void ReturnQuerySubTable(
      typename QuerySubTableLockListType::iterator &query_subtable_lock) {

      // Lock the queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Return and remove it from the locked list.
      (*query_subtable_lock)->Return_(this);
      checked_out_query_subtables_.erase(query_subtable_lock);

      // Check whether the returned query subtable can be exported to
      // other processes subsequently. This can happen when the query
      // subtable has no longer needs additional data to race to the
      // finish line.
      if(! can_be_exported_.back()) {
        can_be_exported_.back() =
          (remaining_work_in_priority_queue_.back() ==
           remaining_work_for_query_subtables_.back()) ;
      }
    }

    /** @brief Locks and checks out a query subtable for a given MPI
     *         process.
     */
    typename QuerySubTableLockListType::iterator LockQuerySubTable(
      int probe_index, int remote_mpi_rank_in) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      checked_out_query_subtables_.push_front(
        boost::intrusive_ptr< QuerySubTableLockType > (
          new QuerySubTableLockType()));
      checked_out_query_subtables_.front()->CheckOut_(
        this, probe_index, remote_mpi_rank_in);
      return checked_out_query_subtables_.begin();
    }

    bool CheckIntegrity(const SubTableIDType &destruct_id) const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      bool flag = false;
      for(unsigned int i = 0; (! flag) && i < query_subtables_.size(); i++) {
        TaskType *it = const_cast<TaskType *>(&(tasks_[i]->top()));
        for(int j = 0; (! flag) && j < tasks_[i]->size(); j++, it++) {
          flag =
            (it->reference_subtable().subtable_id().get<0>() == destruct_id.get<0>() &&
             it->reference_subtable().subtable_id().get<1>() == destruct_id.get<1>() &&
             it->reference_subtable().subtable_id().get<2>() == destruct_id.get<2>());
        }
      }
      for(typename QuerySubTableLockListType::const_iterator it =
            checked_out_query_subtables_.begin();
          (! flag) && it != checked_out_query_subtables_.end(); it++) {
        TaskType *priority_queue_it =
          const_cast<TaskType *>(&((*it)->task_->top()));
        printf("      Reference set: ");
        for(int j = 0; (! flag) && j < (*it)->task_->size(); j++, priority_queue_it++) {
          flag =
            (priority_queue_it->reference_subtable().subtable_id().get<0>() == destruct_id.get<0>() &&
             priority_queue_it->reference_subtable().subtable_id().get<1>() == destruct_id.get<1>() &&
             priority_queue_it->reference_subtable().subtable_id().get<2>() == destruct_id.get<2>());
        }
      }
      return flag;
    }

    /** @brief Prints the current distributed task queue.
     */
    void Print(boost::mpi::communicator &world) const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));

      printf("Distributed queue status:\n");
      printf("  Active query subtables:\n");
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {
        SubTableIDType query_subtable_id = query_subtables_[i]->subtable_id();
        printf("    Query subtable ID: %d %d %d with %d tasks with remaining work %lu originating from %d, currrently holding %lu work in the queue, query result pointer address %p:",
               query_subtable_id.get<0>(), query_subtable_id.get<1>(),
               query_subtable_id.get<2>(),
               static_cast<int>(tasks_[i]->size()),
               remaining_work_for_query_subtables_[i],
               query_subtables_[i]->originating_rank(),
               remaining_work_in_priority_queue_[i],
               query_subtables_[i]->query_result());
        TaskType *it = const_cast<TaskType *>(&(tasks_[i]->top()));
        printf("      Reference set: ");
        for(int j = 0; j < tasks_[i]->size(); j++, it++) {
          printf(" %d %d %d at %d %p, ",
                 it->reference_subtable().subtable_id().get<0>(),
                 it->reference_subtable().subtable_id().get<1>(),
                 it->reference_subtable().subtable_id().get<2>(),
                 it->reference_subtable().cache_block_id(),
                 it->query_subtable().query_result()) ;
        }
        printf("\n");
      }
      printf("  Checked-out query subtables:\n");
      for(typename QuerySubTableLockListType::const_iterator it =
            checked_out_query_subtables_.begin();
          it != checked_out_query_subtables_.end(); it++) {
        SubTableIDType query_subtable_id = (*it)->subtable_id();
        printf(
          "    Query subtable ID: %d %d %d with %d tasks checked out to %d, query result pointer address %p:\n",
          query_subtable_id.get<0>(), query_subtable_id.get<1>(),
          query_subtable_id.get<2>(),
          static_cast<int>((*it)->task_->size()), (*it)->locked_mpi_rank_,
          (*it)->query_subtable_->query_result());
        TaskType *priority_queue_it =
          const_cast<TaskType *>(&((*it)->task_->top()));
        printf("      Reference set: ");
        for(int j = 0; j < (*it)->task_->size(); j++, priority_queue_it++) {
          printf(
            "  %d %d %d at %d %p, ",
            priority_queue_it->reference_subtable().subtable_id().get<0>(),
            priority_queue_it->reference_subtable().subtable_id().get<1>(),
            priority_queue_it->reference_subtable().subtable_id().get<2>(),
            priority_queue_it->reference_subtable().cache_block_id(),
            priority_queue_it->query_subtable().query_result()) ;
        }
        printf("\n");
      }
      table_exchange_.PrintSubTables(world);
      printf("\n");
    }

    /** @brief Initializes a new query subtable queue with its query
     *         subresult.
     */
    int PushNewQueue(
      int originating_rank_in, SubTableType &query_subtable_in) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Get more slots.
      this->GrowSlots_();
      query_subtables_.back()->Alias(query_subtable_in);
      query_subtables_.back()->set_originating_rank(originating_rank_in);
      can_be_exported_.back() = false;
      remaining_work_for_query_subtables_.back() = 0;
      remaining_work_in_priority_queue_.back() = 0;

      // Increment the number of imported subtables.
      num_imported_query_subtables_++;

      // Push in the position for the position that needs to be looked
      // at higher priority.
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

      // Lock the queue.
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
        reference_subtable,
        priority);
      tasks_[ push_index]->push(new_task);

      // Increment the number of tasks.
      num_remaining_tasks_++;

      // Increment the available local computation.
      remaining_local_computation_ += new_task.work();

      // Tally the number or reference points associated with this
      // queue.
      remaining_work_in_priority_queue_[ push_index ] +=
        reference_subtable.start_node()->count();
    }

    /** @brief Returns the subtable stored in the given position of
     *         the cache.
     */
    SubTableType *FindSubTable(int cache_id) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      return table_exchange_.FindSubTable(cache_id);
    }

    /** @brief Pushes a received subtable, locking the cache equal to
     *         the given number of times.
     */
    int push_subtable(
      SubTableType &subtable_in, int num_referenced_as_reference_set) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      return table_exchange_.push_subtable(
               subtable_in, num_referenced_as_reference_set);
    }

    /** @brief Prepares a list of overflowing tasks that are to be
     *         sent to another process.
     */
    template<typename MetricType>
    void PrepareExtraTaskList(
      boost::mpi::communicator &world,
      const MetricType &metric_in,
      int neighbor_rank_in,
      unsigned long int neighbor_remaining_extra_points_to_hold_in,
      const DualtreeLoadBalanceRequestType &neighbor_load_balance_request_in,
      TaskListType *extra_task_list_out) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Loop over every unlocked query subtable and try to pack as
      // many tasks as possible.
      extra_task_list_out->Init(
        world, neighbor_rank_in, neighbor_remaining_extra_points_to_hold_in,
        *this);
      for(int i = 0;
          extra_task_list_out->remaining_extra_points_to_hold() > 0 &&
          i < static_cast<int>(query_subtables_.size()); i++) {

        // The policy is: (1) never to import back again the subtable
        // the current MPI process has exported. The exported query
        // subtables come back to its origin through the flush
        // operation; (2) never to export query subtables that are
        // imported from other MPI processes.
        if((! neighbor_load_balance_request_in.query_subtable_is_owned(
              query_subtables_[i]->subtable_id())) &&
            query_subtables_[i]->table()->rank() == world.rank() &&
            extra_task_list_out->push_back(world, i)) {
          num_exported_query_subtables_++;
          i--;
        }
      }
    }

    /** @brief Sends a load balancing request to the given MPI
     *         process.
     */
    void PrepareLoadBalanceRequest(
      DualtreeLoadBalanceRequestType *load_balance_request) {

      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);
      load_balance_request->Init(
        query_subtables_, checked_out_query_subtables_,
        remaining_local_computation_,
        table_exchange_.remaining_extra_points_to_hold());
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
      can_be_exported_.resize(0);
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

      // Lock the queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Move data around (query subtable flushes, reference subtable
      // forwarding, etc.).
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
            table_exchange_.reference_table()->local_table(),
            table_exchange_.FindByBeginCount(
              reference_begin, reference_count), false);
          alias.set_cache_block_id(cache_id);
          frontier_reference_subtable = &alias;
        }
        boost::tuple<int, int, int> reference_grid(
          frontier_reference_subtable->table()->rank(),
          reference_begin, reference_begin + reference_count);

        // For each query subtree owned by the current process, create
        // a new task if it has not already taken care of the incoming
        // reference table.
        for(int j = 0; j < static_cast<int>(query_subtables_.size()); j++) {
          if(query_subtables_[j]->table()->rank() == world.rank() &&
              assigned_work_[j]->Insert(reference_grid)) {
            this->PushTask(
              world, metric_in, j, * frontier_reference_subtable);
            table_exchange_.LockCache(cache_id, 1);
          }
        } // end of looping over each active query subtable.

        // Also do it for the checked out query subtables.
        for(typename QuerySubTableLockListType::iterator it =
              checked_out_query_subtables_.begin();
            it != checked_out_query_subtables_.end(); it++) {
          if((*it)->Insert_(world, reference_grid)) {
            (*it)->PushTask_(
              this, world, metric_in, * frontier_reference_subtable);
            table_exchange_.LockCache(cache_id, 1);
          }
        } // end of looping over the checked out query subtables.
      } //end of looping over each reference subtree.
    }

    /** @brief Determines whether the MPI process can terminate.
     */
    bool can_terminate() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return (
               remaining_global_computation_ == 0 &&
               num_imported_query_subtables_ == 0 &&
               num_exported_query_subtables_ == 0 &&
               table_exchange_.can_terminate());
    }

    /** @brief Pushes the completed computation for the given query
     *         subtable.
     */
    void push_completed_computation(
      boost::mpi::communicator &comm,
      unsigned long int reference_count_in,
      unsigned long int quantity_in,
      typename QuerySubTableLockListType::iterator &query_subtable_lock) {
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Subtract from the self and queue up a route message.
      remaining_global_computation_ -= quantity_in;
      table_exchange_.push_completed_computation(comm, quantity_in);

      // Update the remaining work for the query tree.
      (*query_subtable_lock)->remaining_work_for_query_subtable_ -=
        reference_count_in;
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

    /** @brief Synchronize with the local MPI result using the given
     *         sub-result.
     */
    void PostComputeSynchronize(
      int num_deterministic_prunes_in,
      int num_probabilistic_prunes_in,
      const QueryResultType &sub_query_results,
      QueryResultType *local_mpi_query_results) {

      // Lock the queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Collect back the result gathered by a task.
      local_mpi_query_results->Accumulate(sub_query_results);

      // Tally up the prune count.
      num_deterministic_prunes_ += num_deterministic_prunes_in;
      num_probabilistic_prunes_ += num_probabilistic_prunes_in;
    }

    /** @brief The constructor.
     */
    DistributedDualtreeTaskQueue() {
      num_deterministic_prunes_ = 0;
      num_exported_query_subtables_ = 0;
      num_imported_query_subtables_ = 0;
      num_probabilistic_prunes_ = 0;
      num_remaining_tasks_ = 0;
      num_threads_ = 1;
      remaining_global_computation_ = 0;
      remaining_local_computation_ = 0;
    }

    int num_deterministic_prunes() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return num_deterministic_prunes_;
    }

    int num_probabilistic_prunes() const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return num_probabilistic_prunes_;
    }

    /** @brief Returns the number of tasks associated with the probing
     *         index.
     */
    int size(int probe_index) const {
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return tasks_[probe_index]->size();
    }

    /** @brief Initializes the task queue.
     */
    void Init(
      boost::mpi::communicator &world,
      int max_subtree_size_in,
      bool do_load_balancing_in,
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
        10 * num_threads_in, &query_subtables_);

      // Initialize the other member variables.
      tasks_.resize(query_subtables_.size());
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {

        // Set up the query subtable.
        query_subtables_[i]->set_query_result(*local_query_result_in);
        query_subtables_[i]->set_cache_block_id(-1);

        // Initialize an empty task priority queue for each query subtable.
        tasks_[i] = boost::intrusive_ptr <
                    TaskPriorityQueueType > (new TaskPriorityQueueType());
      }

      // Initialize the table exchange.
      table_exchange_.Init(
        world, max_subtree_size_in, do_load_balancing_in,
        query_table_in, reference_table_in, this);

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
      remaining_local_computation_ = 0;
      num_remaining_tasks_ = 0;

      // Initialize the completed computation grid for each query tree
      // on this process.
      assigned_work_.resize(query_subtables_.size()) ;
      can_be_exported_.resize(query_subtables_.size()) ;
      remaining_work_for_query_subtables_.resize(query_subtables_.size());
      remaining_work_in_priority_queue_.resize(query_subtables_.size());
      for(unsigned int i = 0; i < query_subtables_.size(); i++) {
        assigned_work_[i] =
          boost::intrusive_ptr <
          core::parallel::DisjointIntIntervals > (
            new core::parallel::DisjointIntIntervals());
        assigned_work_[i]->Init(world);
        can_be_exported_[i] = false;
        remaining_work_for_query_subtables_[i] = total_num_reference_points;
        remaining_work_in_priority_queue_[i] = 0;
      }
    }

    /** @brief Dequeues a task, optionally locking a query subtree
     *         associated with it.
     */
    template<typename MetricType>
    void DequeueTask(
      boost::mpi::communicator &world,
      int thread_id,
      const MetricType &metric_in,
      std::pair<TaskType, int> *task_out,
      typename QuerySubTableLockListType::iterator
      *checked_out_query_subtable) {

      // Lock the task queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // If the number of available task is less than the number of
      // running threads, try to get one.
      //if(static_cast<int>(tasks_.size()) < num_threads_) {
      //this->RedistributeAmongCores_(world, metric_in);
      //}

      // Try to dequeue a task by scanning the list of available query
      // subtables.
      for(int probe_index = 0; task_out->second < 0 &&
          probe_index < static_cast<int>(tasks_.size()); probe_index++) {

        if(this->DequeueTask(
              world, probe_index, task_out, checked_out_query_subtable)) {
          probe_index--;
        }
      }
    }

    /** @brief Examines the top task in the given task list.
     */
    const TaskType &top(int probe_index) const {

      // Lock the task queue.
      core::parallel::scoped_omp_nest_lock lock(
        &(const_cast <
          DistributedDualtreeTaskQueueType * >(this)->task_queue_lock_));
      return tasks_[probe_index]->top();
    }

    /** @brief Removes the top task in the given task list.
     */
    void pop(int probe_index) {

      // Lock the task queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      // Decrement the amount of local computation.
      remaining_local_computation_ -= tasks_[probe_index]->top().work();

      // Decrement the number of reference points associated with the
      // queue.
      remaining_work_in_priority_queue_[ probe_index ] -=
        const_cast< TaskType &>(
          tasks_[
            probe_index ]->top()).reference_subtable().start_node()->count();

      // Pop.
      tasks_[probe_index]->pop();

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
      typename QuerySubTableLockListType::iterator
      *checked_out_query_subtable) {

      // Lock the task queue.
      core::parallel::scoped_omp_nest_lock lock(&task_queue_lock_);

      if(tasks_[probe_index]->size() > 0) {

        // Copy the task and the query subtree number.
        task_out->first = tasks_[ probe_index ]->top();
        task_out->second = probe_index;

        // Pop the task from the priority queue after copying and
        // put a lock on the query subtree.
        tasks_[ probe_index ]->pop();

        // Decrement the number of tasks.
        num_remaining_tasks_--;

        // Decrement the remaining local computation.
        remaining_local_computation_ -= task_out->first.work();

        // Decrement the remaining number of reference points
        // associated with the queue.
        remaining_work_in_priority_queue_[ probe_index ] -=
          task_out->first.reference_subtable().start_node()->count() ;

        // Check out the query subtable completely if requested.
        if(checked_out_query_subtable != NULL) {
          *checked_out_query_subtable =
            this->LockQuerySubTable(probe_index, world.rank());
        }
      }

      // Otherwise, determine whether the cleanup needs to be done.
      else {

        // If the query subtable is on the MPI process of its origin,
        if(query_subtables_[probe_index]->table()->rank() == world.rank()) {
          if(remaining_work_for_query_subtables_[probe_index] == 0) {
            this->Evict_(world, probe_index);
            return true;
          }
        }

        // If the query subtable is not from the MPI process of its
        // origin and it ran out of stuffs to do, flush.
        else if(tasks_[probe_index]->size() == 0) {
          this->Flush_(world, probe_index);
          return true;
        }
      }
      return false;
    }
};
}
}

#endif
