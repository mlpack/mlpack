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
#include "core/parallel/distributed_dualtree_task_queue.h"
#include "core/parallel/distributed_termination.h"
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/parallel/message_tag.h"
#include "core/parallel/table_exchange.h"
#include "core/table/table.h"
#include "core/table/memory_mapped_file.h"

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
  core::parallel::TableExchange < DistributedTableType > &table_exchange,
  const std::vector< boost::tuple<int, int, int, int> > &received_subtable_ids,
  core::parallel::DistributedDualtreeTaskQueue <
  DistributedTableType, FinePriorityQueueType > *distributed_tasks) {

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
    for(int j = 0; j < distributed_tasks->size(); j++) {
      distributed_tasks->PushTask(metric_in, j, reference_table_node_pair);
    }

    // Assuming that each query subtree needs to lock on the reference
    // subtree, do so.
    table_exchange.LockCache(cache_id, distributed_tasks->size());

  } //end of looping over each reference subtree.
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs <
DistributedProblemType >::HashSendList_(
  const std::pair<int, int> &local_rnode_id,
  int query_process_id,
  std::vector <
  core::parallel::SubTableRouteRequest<TableType> > *
  hashed_essential_reference_subtrees) {

  // May consider using a STL map to speed up the hashing.
  int found_index = -1;
  for(unsigned int i = 0;
      i < hashed_essential_reference_subtrees->size(); i++) {

    if((*hashed_essential_reference_subtrees)[i].has_same_subtable_id(local_rnode_id)) {
      found_index = i;
      break;
    }
  }
  if(found_index < 0) {
    hashed_essential_reference_subtrees->resize(
      hashed_essential_reference_subtrees->size() + 1);
    found_index = hashed_essential_reference_subtrees->size() - 1;
    (*hashed_essential_reference_subtrees)[
      found_index].InitSubTableForSending(
        reference_table_->local_table(), local_rnode_id);
  }
  (*hashed_essential_reference_subtrees)[
    found_index].add_destination(query_process_id);
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
  core::parallel::SubTableRouteRequest<TableType> > *hashed_essential_reference_subtrees,
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
      std::pair<int, int> local_rnode_id(
        local_reference_node->begin(), local_reference_node->count());
      while(qnode_it.HasNext()) {
        int query_process_id;
        qnode_it.Next(&query_process_id);
        (*essential_reference_subtrees)[
          query_process_id].push_back(local_rnode_id);

        // Add the query process ID to the list of query processes
        // that this reference subtree needs to be sent to.
        HashSendList_(
          local_rnode_id, query_process_id,
          hashed_essential_reference_subtrees);

        // Push in the squared distance range.
        (*squared_distance_ranges)[query_process_id].push_back(
          squared_distance_range);
      }
    }
    else {

      ComputeEssentialReferenceSubtrees_(
        metric_in, max_reference_subtree_size,
        global_query_node, local_reference_node->left(),
        essential_reference_subtrees, hashed_essential_reference_subtrees,
        squared_distance_ranges, extrinsic_prunes);
      ComputeEssentialReferenceSubtrees_(
        metric_in, max_reference_subtree_size,
        global_query_node, local_reference_node->right(),
        essential_reference_subtrees, hashed_essential_reference_subtrees,
        squared_distance_ranges, extrinsic_prunes);
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
      essential_reference_subtrees, hashed_essential_reference_subtrees,
      squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node,
      essential_reference_subtrees, hashed_essential_reference_subtrees,
      squared_distance_ranges, extrinsic_prunes);
  }
  else {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->left(),
      essential_reference_subtrees, hashed_essential_reference_subtrees,
      squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->right(),
      essential_reference_subtrees, hashed_essential_reference_subtrees,
      squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->left(),
      essential_reference_subtrees, hashed_essential_reference_subtrees,
      squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->right(),
      essential_reference_subtrees, hashed_essential_reference_subtrees,
      squared_distance_ranges, extrinsic_prunes);
  }
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs <
DistributedProblemType >::InitialSetup_(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results,
  core::parallel::TableExchange < DistributedTableType > &table_exchange,
  std::vector< std::vector< std::pair<int, int> > > *
  essential_reference_subtrees_to_send,
  std::vector< std::vector< core::math::Range > > *send_priorities,
  std::vector <
  core::parallel::SubTableRouteRequest<TableType> >
  *hashed_essential_reference_subtress_to_send,
  std::vector< SendRequestPriorityQueueType > *prioritized_send_subtables,
  int *num_reference_subtrees_to_send,
  std::vector< std::vector< std::pair<int, int> > > *reference_frontier_lists,
  std::vector< std::vector< core::math::Range > > *receive_priorities,
  int *num_reference_subtrees_to_receive,
  core::parallel::DistributedDualtreeTaskQueue <
  DistributedTableType, FinePriorityQueueType > *distributed_tasks) {

  // The max number of points for the reference subtree for each task.
  int max_reference_subtree_size = max_subtree_size_;

  // For each process, initialize the distributed task object.
  distributed_tasks->Init(
    query_table_->local_table(), omp_get_max_threads(), table_exchange);

  // Each process needs to customize its reference set for each
  // participating query process.
  std::vector<double> extrinsic_prunes_broadcast(world_->size(), 0.0);
  ComputeEssentialReferenceSubtrees_(
    metric, max_reference_subtree_size, query_table_->get_tree(),
    reference_table_->local_table()->get_tree(),
    essential_reference_subtrees_to_send,
    hashed_essential_reference_subtress_to_send,
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

        // Reference subtables on the self are already available.
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
  GenerateTasks_(
    metric, table_exchange, received_subtable_ids, distributed_tasks);

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
  for(int i = 0; i < static_cast<int>(reference_frontier_lists->size()); i++) {

    // Exclude the reference subtrees from the self.
    if(i != world_->rank()) {
      (*num_reference_subtrees_to_receive) +=
        ((*reference_frontier_lists)[i]).size();
    }
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
  core::parallel::DistributedDualtreeTaskQueue <
  DistributedTableType, FinePriorityQueueType > distributed_tasks;

  // An abstract way of collaborative subtable exchanges.
  core::parallel::TableExchange < DistributedTableType > table_exchange;
  table_exchange.Init(
    *world_, *(reference_table_->local_table()),
    max_num_work_to_dequeue_per_stage_);

  // The number of reference subtrees to receive and to send in total.
  int num_reference_subtrees_to_send;
  int num_reference_subtrees_to_receive;
  std::vector <
  core::parallel::SubTableRouteRequest<TableType> >
  hashed_essential_reference_subtrees_to_send;
  InitialSetup_(
    metric, query_results, table_exchange,
    &reference_subtrees_to_send, &send_priorities,
    &hashed_essential_reference_subtrees_to_send,
    &prioritized_send_subtables,
    &num_reference_subtrees_to_send,
    &reference_subtrees_to_receive,
    &receive_priorities,
    &num_reference_subtrees_to_receive,
    &distributed_tasks);

  // The number of completed sends used for determining the
  // termination condition.
  int num_completed_sends = 0;

  // Used for termination condition check.
  core::parallel::DistributedTermination termination_check;
  termination_check.Init(*world_);

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
            metric, table_exchange, received_subtable_ids, &distributed_tasks);
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
        for(int i = 0; i < distributed_tasks.size(); i++) {

          // Index to probe.
          int probe_index = (thread_id + i) % distributed_tasks.size();

#pragma omp critical
          {
            distributed_tasks.DequeueTask(probe_index, &found_task, true);
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
          TableType *task_reference_table =
            found_task.first.reference_table();
          TreeType *task_starting_rnode =
            found_task.first.reference_start_node();
          int task_reference_cache_id = found_task.first.cache_id();

          // Initialize the argument, the problem, and the engine.
          sub_argument.Init(
            task_reference_table,
            query_table_->local_table(), problem_->global());
          sub_problem.Init(sub_argument, &(problem_->global()));
          sub_engine.Init(sub_problem);

          // Set the starting query node.
          sub_engine.set_query_start_node(found_task.first.query_start_node());

          // Set the starting reference node.
          sub_engine.set_reference_start_node(task_starting_rnode);

          // Fire away the computation.
          sub_engine.Compute(metric, query_results, false);

#pragma omp critical
          {
            num_deterministic_prunes_ += sub_engine.num_deterministic_prunes();
            num_probabilistic_prunes_ += sub_engine.num_probabilistic_prunes();

            // After finishing, the lock on the query subtree is released.
            distributed_tasks.UnlockQuerySubtree(metric, found_task.second);

            // Release the reference subtable.
            table_exchange.ReleaseCache(task_reference_cache_id);

          } // end of a critical section.

        } // end of finding a task.
        else {

#pragma omp critical
          {
            // Otherwise, ask other threads to share the work.
            distributed_tasks.set_split_subtree_flag();
          }
        }
      } // end of attempting to deque a task.

      // Quit if all of the sending is done and the task queue is
      // empty.
#pragma omp critical
      {
        work_left_to_do =
          !(num_reference_subtrees_to_receive ==
            table_exchange.total_num_subtables_received() &&
            num_reference_subtrees_to_send == num_completed_sends &&
            distributed_tasks.is_empty());
      } // end of a critical section.
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

  // The MPI timer.
  boost::mpi::timer timer;

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
  std::cerr << "Process " << world_->rank() << " took " <<
            timer.elapsed() << " seconds to compute.\n";
  std::cerr << "  Deterministic prunes: " <<
            this->num_deterministic_prunes() <<
            " , probabilistic prunes: " <<
            this->num_probabilistic_prunes() << "\n";
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
