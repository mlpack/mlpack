/** @file distributed_dualtree_dfs_dev.h
 *
 *  The generic algorithm template for distributed dual-tree
 *  computation.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_DFS_DEV_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_DFS_DEV_H

#include <boost/bind.hpp>
#include <boost/mpi.hpp>
#include <boost/tuple/tuple.hpp>
#include <map>
#include <queue>
#include "core/parallel/distributed_dualtree_dfs.h"
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/parallel/dualtree_load_balancer.h"
#include "core/parallel/table_exchange.h"
#include "core/table/table.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace parallel {

template<typename ProblemType>
int DistributedDualtreeDfs<ProblemType>::num_deterministic_prunes() const {
  return num_deterministic_prunes_;
}

template<typename ProblemType>
int DistributedDualtreeDfs<ProblemType>::num_probabilistic_prunes() const {
  return num_probabilistic_prunes_;
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs <
DistributedProblemType >::GenerateTasks_(
  const MetricType &metric_in,
  core::parallel::TableExchange <
  DistributedTableType, SubTableType > &table_exchange,
  std::vector<TreeType *> &local_query_subtrees,
  const std::vector< boost::tuple<int, int, int, int> > &received_subtable_ids,
  std::vector< FinePriorityQueueType > *tasks) {

  for(unsigned int i = 0; i < received_subtable_ids.size(); i++) {

    // Find the reference process ID and grab its subtable.
    int reference_begin = received_subtable_ids[i].get<1>();
    int reference_count = received_subtable_ids[i].get<2>();
    int cache_id = received_subtable_ids[i].get<3>();
    SubTableType *frontier_reference_subtable =
      table_exchange.FindSubTable(cache_id);

    // Find the table and the starting reference node.
    TableType *frontier_reference_table =
      (frontier_reference_subtable != NULL) ?
      frontier_reference_subtable->table() : reference_table_->local_table();
    TreeType *reference_starting_node =
      (frontier_reference_subtable != NULL) ?
      frontier_reference_subtable->table()->get_tree() :
      reference_table_->local_table()->
      get_tree()->FindByBeginCount(
        reference_begin, reference_count);
    boost::tuple<TableType *, TreeType *, int> reference_table_node_pair(
      frontier_reference_table, reference_starting_node, cache_id);

    // For each query subtree, create a new task.
    for(unsigned int j = 0; j < local_query_subtrees.size(); j++) {

      // Create a fine frontier object to be dequeued by each thread
      // later.
      core::math::Range squared_distance_range(
        local_query_subtrees[j]->bound().RangeDistanceSq(
          metric_in, reference_table_node_pair.get<1>()->bound()));
      FineFrontierObjectType new_task =
        boost::tuple < TreeType * ,
        boost::tuple<TableType *, TreeType *, int>, double > (
          local_query_subtrees[j], reference_table_node_pair,
          - squared_distance_range.mid());
      (*tasks)[j].push(new_task);
    }

    // Assuming that each query subtree needs to lock on the reference
    // subtree, do so.
    table_exchange.LockCache(cache_id, local_query_subtrees.size());
  } //end of looping over each reference subtree.
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs <
DistributedProblemType >::ComputeEssentialReferenceSubtrees_(
  const MetricType &metric_in,
  int max_reference_subtree_size,
  DistributedTreeType *global_query_node, TreeType *local_reference_node,
  std::vector <
  std::vector< std::pair<int, int> > > *essential_reference_subtrees,
  std::vector <
  std::vector< core::math::Range > > *squared_distance_ranges,
  std::vector< double > *extrinsic_prunes) {

  // Compute the squared distance ranges between the query node and
  // the reference node.
  core::math::Range squared_distance_range =
    global_query_node->bound().RangeDistanceSq(
      metric_in, local_reference_node->bound());

  // If the pair is prunable, then return.
  if(problem_->global().ConsiderExtrinsicPrune(squared_distance_range)) {
    typename TableType::TreeIterator qnode_it =
      query_table_->get_node_iterator(global_query_node);
    while(qnode_it.HasNext()) {
      int query_process_id;
      qnode_it.Next(&query_process_id);
      (*extrinsic_prunes)[query_process_id] += local_reference_node->count();
    }
    return;
  }

  if(global_query_node->is_leaf()) {

    // If the reference node size is within the size limit, then for
    // each query process in the set, add the reference node to the
    // list.
    if(local_reference_node->count() <= max_reference_subtree_size ||
        local_reference_node->is_leaf()) {
      typename TableType::TreeIterator qnode_it =
        query_table_->get_node_iterator(global_query_node);
      while(qnode_it.HasNext()) {
        int query_process_id;
        qnode_it.Next(&query_process_id);
        (*essential_reference_subtrees)[
          query_process_id].push_back(
            std::pair<int, int>(
              local_reference_node->begin(), local_reference_node->count()));
        (*squared_distance_ranges)[query_process_id].push_back(
          squared_distance_range);
      }
    }
    else {

      ComputeEssentialReferenceSubtrees_(
        metric_in, max_reference_subtree_size,
        global_query_node, local_reference_node->left(),
        essential_reference_subtrees, squared_distance_ranges,
        extrinsic_prunes);
      ComputeEssentialReferenceSubtrees_(
        metric_in, max_reference_subtree_size,
        global_query_node, local_reference_node->right(),
        essential_reference_subtrees, squared_distance_ranges,
        extrinsic_prunes);
    }
    return;
  }

  // Here, we know that the global query node is a non-leaf, so we
  // need to split both ways.
  if(local_reference_node->count() <= max_reference_subtree_size ||
      local_reference_node->is_leaf()) {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node,
      essential_reference_subtrees, squared_distance_ranges,
      extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node,
      essential_reference_subtrees, squared_distance_ranges,
      extrinsic_prunes);
  }
  else {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->left(),
      essential_reference_subtrees, squared_distance_ranges,
      extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->right(),
      essential_reference_subtrees, squared_distance_ranges,
      extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->left(),
      essential_reference_subtrees, squared_distance_ranges,
      extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->right(),
      essential_reference_subtrees, squared_distance_ranges,
      extrinsic_prunes);
  }
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::RedistributeQuerySubtrees_(
  const std::vector<TreeType *> &local_query_subtrees,
  const std::vector<int> &local_query_subtree_assignments,
  int total_num_query_subtrees_to_receive,
  core::parallel::TableExchange <
  DistributedTableType, SubTableType > *query_subtree_cache) {

  // Initialize the query subtree cache.
  query_subtree_cache->Init(
    * world_, * query_table_->local_table(),
    max_num_work_to_dequeue_per_stage_);

  // Fill out the query subtree send requests.
  std::vector <
  SendRequestPriorityQueueType > query_subtree_send_requests(world_->size());
  for(unsigned int i = 0; i < local_query_subtree_assignments.size(); i++) {
    query_subtree_send_requests[local_query_subtree_assignments[i]].push(
      core::parallel::SubTableSendRequest(
        local_query_subtree_assignments[i],
        local_query_subtrees[i]->begin(),
        local_query_subtrees[i]->count(), 0.0));
  }

  // Exchange until done.
  int num_completed_sends = 0;
  int num_completed_receives = 0;
  do {
    std::vector< boost::tuple<int, int, int, int> > received_query_subtable_ids;
    query_subtree_cache->AsynchSendReceive(
      * world_, query_subtree_send_requests, &received_query_subtable_ids,
      &num_completed_sends);
    num_completed_receives += received_query_subtable_ids.size();

  }
  while(
    num_completed_sends <
    static_cast<int>(local_query_subtree_assignments.size()) ||
    num_completed_receives < total_num_query_subtrees_to_receive);
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs <
DistributedProblemType >::InitialSetup_(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results,
  core::parallel::TableExchange <
  DistributedTableType, SubTableType > &table_exchange,
  std::vector< TreeType *> *local_query_subtrees,
  std::vector< std::vector< std::pair<int, int> > > *
  essential_reference_subtrees_to_send,
  std::vector< std::vector< core::math::Range > > *send_priorities,
  std::vector< SendRequestPriorityQueueType > *prioritized_send_subtables,
  int *num_reference_subtrees_to_send,
  std::vector< std::vector< std::pair<int, int> > > *reference_frontier_lists,
  std::vector< std::vector< core::math::Range > > *receive_priorities,
  int *num_reference_subtrees_to_receive,
  std::vector< FinePriorityQueueType > *tasks) {

  // The max number of points for the query subtree for each task.
  int max_query_subtree_size = max_subtree_size_;

  // The max number of points for the reference subtree for each task.
  int max_reference_subtree_size = max_subtree_size_;

  // For each process, break up the local query tree into a list of
  // subtree query lists.
  query_table_->local_table()->get_frontier_nodes(
    max_query_subtree_size, local_query_subtrees);

  // Each process needs to customize its reference set for each
  // participating query process.
  std::vector<double> extrinsic_prunes_broadcast(world_->size(), 0.0);
  ComputeEssentialReferenceSubtrees_(
    metric, max_reference_subtree_size, query_table_->get_tree(),
    reference_table_->local_table()->get_tree(),
    essential_reference_subtrees_to_send,
    send_priorities, &extrinsic_prunes_broadcast);

  // Fill out the prioritized send list.
  std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;
  *num_reference_subtrees_to_send = 0;
  for(int i = 0; i < world_->size(); i++) {
    for(unsigned int j = 0;
        j < (*essential_reference_subtrees_to_send)[i].size(); j++) {

      int reference_begin =
        (*essential_reference_subtrees_to_send)[i][j].first;
      int reference_count =
        (*essential_reference_subtrees_to_send)[i][j].second;
      if(i == world_->rank()) {
        received_subtable_ids.push_back(
          boost::make_tuple(i, reference_begin, reference_count, -1));
      }
      else {
        (*prioritized_send_subtables)[i].push(
          core::parallel::SubTableSendRequest(
            i, reference_begin, reference_count,
            - (*send_priorities)[i][j].mid()));

        // Increment the number of subtrees to send.
        (*num_reference_subtrees_to_send)++;
      }
    }
  }

  // Fill out the initial task consisting of the reference trees on
  // the same process.
  tasks->resize(local_query_subtrees->size());
  GenerateTasks_(
    metric, table_exchange, * local_query_subtrees,
    received_subtable_ids, tasks);

  // Do an all to all to let each participating query process its
  // initial frontier.
  std::vector< double > extrinsic_prune_lists;
  boost::mpi::all_to_all(
    *world_, *essential_reference_subtrees_to_send, *reference_frontier_lists);
  boost::mpi::all_to_all(
    *world_, *send_priorities, *receive_priorities);
  boost::mpi::all_to_all(
    *world_, extrinsic_prunes_broadcast, extrinsic_prune_lists);

  // Tally up the number of reference subtrees to receive for the
  // current MPI process. Tally up the number of pruned reference
  // subtrees.
  *num_reference_subtrees_to_receive = 0;
  for(unsigned int i = 0; i < reference_frontier_lists->size(); i++) {
    (*num_reference_subtrees_to_receive) +=
      ((*reference_frontier_lists)[i]).size();
  }

  // Add up the initial pruned amounts and reseed it on the query
  // side.
  double initial_pruned =
    std::accumulate(
      extrinsic_prune_lists.begin(), extrinsic_prune_lists.end(), 0.0) -
    extrinsic_prune_lists[world_->rank()];
  core::gnp::DualtreeDfs<ProblemType>::PreProcess(
    query_table_->local_table(), query_table_->local_table()->get_tree(),
    query_results, initial_pruned);

  // Load-balance the query subtrees.
  int total_num_query_subtrees_to_receive;
  core::parallel::TableExchange <
  DistributedTableType, SubTableType > query_subtree_cache;
  std::vector<int> local_query_subtree_assignments;
  core::parallel::DualtreeLoadBalancer::Compute(
    *world_,
    *local_query_subtrees,
    *essential_reference_subtrees_to_send,
    *reference_frontier_lists,
    *num_reference_subtrees_to_receive,
    &local_query_subtree_assignments,
    &total_num_query_subtrees_to_receive);

  // Re-distribute the query subtrees based on the assignments.
  RedistributeQuerySubtrees_(
    *local_query_subtrees, local_query_subtree_assignments,
    total_num_query_subtrees_to_receive, & query_subtree_cache);
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs<DistributedProblemType>::AllToAllIReduce_(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // Figure out the list of reference subtrees to send and to receive.
  std::vector< std::vector< std::pair<int, int> > >
  reference_subtrees_to_send(world_->size());
  std::vector <
  std::vector< core::math::Range> > send_priorities(world_->size());
  std::vector <
  SendRequestPriorityQueueType > prioritized_send_subtables(world_->size());
  std::vector <
  std::vector< std::pair<int, int> > > reference_subtrees_to_receive;
  std::vector< std::vector< core::math::Range> > receive_priorities;

  // The list of prioritized tasks this MPI process needs to take care
  // of.
  std::vector< FinePriorityQueueType > tasks;

  // An abstract way of collaborative subtable exchanges.
  core::parallel::TableExchange <
  DistributedTableType, SubTableType > table_exchange;
  table_exchange.Init(
    *world_, *(reference_table_->local_table()),
    max_num_work_to_dequeue_per_stage_);

  // The local query subtrees.
  std::vector< TreeType *> local_query_subtrees;

  // The number of reference subtrees to receive and to send in total.
  int num_reference_subtrees_to_send;
  int num_reference_subtrees_to_receive;
  InitialSetup_(
    metric, query_results, table_exchange, &local_query_subtrees,
    &reference_subtrees_to_send, &send_priorities,
    &prioritized_send_subtables,
    &num_reference_subtrees_to_send,
    &reference_subtrees_to_receive,
    &receive_priorities,
    &num_reference_subtrees_to_receive,
    &tasks);

  // The number of reference subtree releases (used for termination
  // condition) for the current MPI process.
  int num_reference_subtree_releases = 0;

  // The global list of query subtree that is being computed. This is
  // necessary for preventing two threads from grabbing tasks that
  // involve the same query subtree.
  std::deque<bool> active_query_subtrees(local_query_subtrees.size(), false);

  // The number of completed sends used for determining the
  // termination condition.
  int num_completed_sends = 0;

  // OpenMP parallel region. The master thread is the only one that is
  // allowed to make MPI calls (sending and receiving reference
  // subtables).
#pragma omp parallel
  {

    // The thread ID.
    int thread_id = omp_get_thread_num();

    // Used for determining the termination condition.
    bool work_left_to_do = true;

    do {

      std::pair<FineFrontierObjectType, int> found_task;
      found_task.second = -1;

      // The master thread routine.
      if(thread_id == 0) {

#pragma omp critical
        {
          std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;
          table_exchange.AsynchSendReceive(
            *world_, prioritized_send_subtables, &received_subtable_ids,
            &num_completed_sends);

          // Generate the list of work and put it into the queue.
          GenerateTasks_(
            metric, table_exchange, local_query_subtrees,
            received_subtable_ids, &tasks);
        }
      } // end of the master thread.

      // After enqueing, everyone else tries to dequeue the tasks. The
      // master only dequeues only if it is the only one running or it
      // has sent everything to every process.
      bool quick_test = false;
#pragma omp critical
      {
        quick_test = (num_reference_subtrees_to_send == num_completed_sends);
      }
      if(thread_id > 0 || omp_get_num_threads() == 1 || quick_test) {
        bool all_empty = true;
        for(unsigned int i = 0; i < tasks.size(); i++) {

          // Index to probe.
          int probe_index = (thread_id + i) % tasks.size();

#pragma omp critical
          {

            // Check whether the current query subtree is empty.
            all_empty = all_empty && (tasks[ probe_index ].size() == 0);

            // Try to see if the thread can dequeue a task here.
            if((! all_empty) && (! active_query_subtrees[ probe_index ]) &&
                tasks[ probe_index ].size() > 0) {

              // Copy the task and the query subtree number.
              found_task.first = tasks[ probe_index ].top();
              found_task.second = probe_index;

              // Pop the task from the priority queue after copying and
              // put a lock on the query subtree.
              tasks[ probe_index ].pop();
              active_query_subtrees[ probe_index ] = true;
            }
          } // end of pragma omp critical

          if(found_task.second >= 0) {

            // If something is found, then break.
            break;
          }
        } // end of for-loop.

        // If found something to run on, then call the serial dual-tree
        // method.
        if(found_task.second >= 0) {

          // Run a sub-dualtree algorithm on the computation object.
          core::gnp::DualtreeDfs<ProblemType> sub_engine;
          ProblemType sub_problem;
          ArgumentType sub_argument;
          TableType *task_reference_table = found_task.first.get<1>().get<0>();
          TreeType *task_starting_rnode = found_task.first.get<1>().get<1>();
          int task_reference_cache_id = found_task.first.get<1>().get<2>();

          // Initialize the argument, the problem, and the engine.
          sub_argument.Init(
            task_reference_table,
            query_table_->local_table(), problem_->global());
          sub_problem.Init(sub_argument, &(problem_->global()));
          sub_engine.Init(sub_problem);

          // Set the starting query node.
          sub_engine.set_query_start_node(found_task.first.get<0>());

          // Set the starting reference node.
          sub_engine.set_reference_start_node(task_starting_rnode);

          // Fire away the computation.
          sub_engine.Compute(metric, query_results, false);

#pragma omp critical
          {
            num_deterministic_prunes_ += sub_engine.num_deterministic_prunes();
            num_probabilistic_prunes_ += sub_engine.num_probabilistic_prunes();

            // After finishing, the lock on the query subtree is released.
            active_query_subtrees[ found_task.second ] = false;

            // Count the number of times the reference subtree is released.
            num_reference_subtree_releases++;

            // Release the reference subtable.
            table_exchange.ReleaseCache(task_reference_cache_id);
          }
        } // end of finding a task.

        // Quit if all of the sending is done and the task queue is
        // empty.
#pragma omp critical
        {
          work_left_to_do =
            !(table_exchange.is_empty() &&
              num_reference_subtree_releases ==
              static_cast<int>(
                local_query_subtrees.size() *
                num_reference_subtrees_to_receive) &&
              num_reference_subtrees_to_send == num_completed_sends);
        }
      }
    }
    while(work_left_to_do);

  } // end of omp parallel
}

template<typename DistributedProblemType>
DistributedProblemType *DistributedDualtreeDfs <
DistributedProblemType >::problem() {
  return problem_;
}

template<typename DistributedProblemType>
typename DistributedProblemType::DistributedTableType *
DistributedDualtreeDfs<DistributedProblemType>::query_table() {
  return query_table_;
}

template<typename DistributedProblemType>
typename DistributedProblemType::DistributedTableType *
DistributedDualtreeDfs<DistributedProblemType>::reference_table() {
  return reference_table_;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::ResetStatistic() {
  ResetStatisticRecursion_(query_table_->get_tree(), query_table_);
}

template<typename DistributedProblemType>
DistributedDualtreeDfs<DistributedProblemType>::DistributedDualtreeDfs() {
  leaf_size_ = 0;
  max_subtree_size_ = 20000;
  max_num_work_to_dequeue_per_stage_ = 5;
  max_computation_frontier_size_ = 0;
  num_deterministic_prunes_ = 0;
  num_probabilistic_prunes_ = 0;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs<DistributedProblemType>::set_work_params(
  int leaf_size_in,
  int max_subtree_size_in,
  int max_num_work_to_dequeue_per_stage_in) {

  leaf_size_ = leaf_size_in;
  max_subtree_size_ = max_subtree_size_in;
  max_num_work_to_dequeue_per_stage_ = max_num_work_to_dequeue_per_stage_in;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs<DistributedProblemType>::Init(
  boost::mpi::communicator *world_in,
  DistributedProblemType &problem_in) {
  world_ = world_in;
  problem_ = &problem_in;
  query_table_ = problem_->query_table();
  reference_table_ = problem_->reference_table();
  ResetStatistic();

  if(query_table_ != reference_table_) {
    ResetStatisticRecursion_(reference_table_->get_tree(), reference_table_);
  }
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs<DistributedProblemType>::Compute(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // Allocate space for storing the final results.
  query_results->Init(problem_->global(), query_table_->n_entries());

  // Preprocess the global query tree and the local query tree owned
  // by each process.
  PreProcess_(query_table_->get_tree());
  PreProcess_(query_table_->local_table()->get_tree());

  // Preprocess the global reference tree, and the local reference
  // tree owned by each process. This part needs to be fixed so that
  // it does a true bottom-up refinement using an MPI-gather.
  core::gnp::DualtreeDfs<ProblemType> self_engine;
  ProblemType self_problem;
  ArgumentType self_argument;
  self_argument.Init(problem_->global());
  self_problem.Init(self_argument, &(problem_->global()));
  self_engine.Init(self_problem);
  core::gnp::DualtreeDfs<ProblemType>::PreProcessReferenceTree(
    self_engine.problem()->global(),
    reference_table_->local_table()->get_tree());
  PreProcessReferenceTree_(reference_table_->get_tree());

  // Figure out each process's work using the global tree. a 2D matrix
  // workspace. This is currently doing an all-reduce type of
  // exchange.
  AllToAllIReduce_(metric, query_results);
  world_->barrier();
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::ResetStatisticRecursion_(
  typename DistributedProblemType::DistributedTableType::TreeType *node,
  typename DistributedProblemType::DistributedTableType * table) {
  node->stat().SetZero();
  if(node->is_leaf() == false) {
    ResetStatisticRecursion_(node->left(), table);
    ResetStatisticRecursion_(node->right(), table);
  }
}

template<typename DistributedProblemType>
template<typename TemplateTreeType>
void DistributedDualtreeDfs <
DistributedProblemType >::PreProcessReferenceTree_(
  TemplateTreeType *rnode) {

  // Does not do anything yet.
}

template<typename DistributedProblemType>
template<typename TemplateTreeType>
void DistributedDualtreeDfs<DistributedProblemType>::PreProcess_(
  TemplateTreeType *qnode) {

  typename DistributedProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.SetZero();

  if(! qnode->is_leaf()) {
    PreProcess_(qnode->left());
    PreProcess_(qnode->right());
  }
}
}
}

#endif
