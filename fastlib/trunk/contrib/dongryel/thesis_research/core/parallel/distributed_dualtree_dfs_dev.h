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
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/parallel/message_tag.h"
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
void DistributedDualtreeDfs <
DistributedProblemType >::HashSendList_(
  const std::pair<int, int> &local_rnode_id,
  int query_process_id,
  std::vector <
  core::parallel::RouteRequest<SubTableType> > *
  hashed_essential_reference_subtrees) {

  // May consider using a STL map to speed up the hashing.
  int found_index = -1;
  for(unsigned int i = 0;
      i < hashed_essential_reference_subtrees->size(); i++) {

    if((*hashed_essential_reference_subtrees)[i].object().has_same_subtable_id(
          local_rnode_id)) {
      found_index = i;
      break;
    }
  }
  if(found_index < 0) {
    hashed_essential_reference_subtrees->resize(
      hashed_essential_reference_subtrees->size() + 1);
    found_index = hashed_essential_reference_subtrees->size() - 1;
    (*hashed_essential_reference_subtrees)[ found_index ].Init(*world_);
    (*hashed_essential_reference_subtrees)[
      found_index].object().Init(
        reference_table_->local_table(),
        reference_table_->local_table()->get_tree()->FindByBeginCount(
          local_rnode_id.first, local_rnode_id.second), false);
    (*hashed_essential_reference_subtrees)[
      found_index].set_object_is_valid_flag(true);
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
  core::parallel::RouteRequest<SubTableType> > *hashed_essential_reference_subtrees,
  std::vector <
  std::vector< core::math::Range > > *squared_distance_ranges,
  std::vector< unsigned long int > *extrinsic_prunes) {

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
        if(query_process_id != world_->rank()) {
          HashSendList_(
            local_rnode_id, query_process_id,
            hashed_essential_reference_subtrees);
        }

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
  std::vector< std::vector< std::pair<int, int> > > *
  essential_reference_subtrees_to_send,
  std::vector< std::vector< core::math::Range > > *send_priorities,
  std::vector <
  core::parallel::RouteRequest<SubTableType> >
  *hashed_essential_reference_subtress_to_send,
  std::vector< std::vector< std::pair<int, int> > > *reference_frontier_lists,
  std::vector< std::vector< core::math::Range > > *receive_priorities,
  core::parallel::DistributedDualtreeTaskQueue <
  DistributedTableType,
  FinePriorityQueueType, ResultType > *distributed_tasks) {

  // The max number of points for the reference subtree for each task.
  int max_reference_subtree_size = max_subtree_size_;

  // For each process, initialize the distributed task object.
  distributed_tasks->Init(
    *world_, max_subtree_size_, query_table_, reference_table_, query_results,
    omp_get_max_threads());

  // Each process needs to customize its reference set for each
  // participating query process.
  std::vector<unsigned long int> extrinsic_prunes_broadcast(world_->size(), 0);
  ComputeEssentialReferenceSubtrees_(
    metric, max_reference_subtree_size, query_table_->get_tree(),
    reference_table_->local_table()->get_tree(),
    essential_reference_subtrees_to_send,
    hashed_essential_reference_subtress_to_send,
    send_priorities, &extrinsic_prunes_broadcast);

  // Fill out the prioritized send list.
  std::vector< boost::tuple<int, int, int, int> > received_subtable_ids;
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
    }
  }

  // Fill out the initial task consisting of the reference trees on
  // the same process.
  distributed_tasks->GenerateTasks(* world_, metric, received_subtable_ids);

  // Do an all to all to let each participating query process its
  // initial frontier.
  std::vector< unsigned long int > extrinsic_prune_lists;
  boost::mpi::all_to_all(
    *world_, *essential_reference_subtrees_to_send, *reference_frontier_lists);
  boost::mpi::all_to_all(
    *world_, *send_priorities, *receive_priorities);
  boost::mpi::all_to_all(
    *world_, extrinsic_prunes_broadcast, extrinsic_prune_lists);

  // Add up the initial pruned amounts and reseed it on the query
  // side.
  unsigned long int initial_pruned =
    std::accumulate(
      extrinsic_prune_lists.begin(), extrinsic_prune_lists.end(), 0);
  double initial_pruned_cast =
    static_cast<double>(initial_pruned);
  core::gnp::DualtreeDfs<ProblemType>::PreProcess(
    query_table_->local_table(), query_table_->local_table()->get_tree(),
    query_results, initial_pruned_cast);

  // Also seed it on the distributed termination checker.
  unsigned long int initial_completed_work =
    static_cast<unsigned long int>(query_table_->local_table()->n_entries()) *
    initial_pruned;
  distributed_tasks->push_completed_computation(
    *world_, initial_pruned, initial_completed_work);
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
  std::vector< std::pair<int, int> > > reference_subtrees_to_receive;
  std::vector< std::vector< core::math::Range> > receive_priorities;

  // The list of prioritized tasks this MPI process needs to take care
  // of.
  core::parallel::DistributedDualtreeTaskQueue <
  DistributedTableType, FinePriorityQueueType, ResultType > distributed_tasks;

  // The number of reference subtrees to receive and to send in total.
  std::vector <
  core::parallel::RouteRequest<SubTableType> >
  hashed_essential_reference_subtrees_to_send;
  InitialSetup_(
    metric, query_results, &reference_subtrees_to_send, &send_priorities,
    &hashed_essential_reference_subtrees_to_send,
    &reference_subtrees_to_receive,
    &receive_priorities,
    &distributed_tasks);

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

      // The task found in the current iteration.
      std::pair<FineFrontierObjectType, int> found_task;
      found_task.second = -1;

      // Only the master thread makes MPI calls.
      if(thread_id == 0) {
        distributed_tasks.SendReceive(
          metric, *world_, hashed_essential_reference_subtrees_to_send);
      }

      // After enqueing, everyone else tries to dequeue the tasks.
      distributed_tasks.DequeueTask(*world_, metric, &found_task, true);

      // If found something to run on, then call the serial dual-tree
      // method.
      if(found_task.second >= 0) {

        // Run a sub-dualtree algorithm on the computation object.
        core::gnp::DualtreeDfs<ProblemType> sub_engine;
        ProblemType sub_problem;
        ArgumentType sub_argument;
        TableType *task_query_table =
          found_task.first.query_subtable().table();
        TableType *task_reference_table =
          found_task.first.reference_subtable().table();
        TreeType *task_starting_rnode =
          found_task.first.reference_start_node();
        int task_reference_cache_id =
          found_task.first.reference_subtable_cache_block_id();

        // Initialize the argument, the problem, and the engine.
        sub_argument.Init(
          task_reference_table, task_query_table, problem_->global());
        sub_problem.Init(sub_argument, &(problem_->global()));
        sub_engine.Init(sub_problem);

        // Set the starting query node.
        sub_engine.set_query_start_node(
          found_task.first.query_start_node());

        // Set the starting reference node.
        sub_engine.set_reference_start_node(task_starting_rnode);

        // Fire away the computation.
        sub_engine.Compute(
          metric, found_task.first.query_result(), false);

#pragma omp critical
        {
          num_deterministic_prunes_ += sub_engine.num_deterministic_prunes();
          num_probabilistic_prunes_ += sub_engine.num_probabilistic_prunes();
        }

        // Push in the completed amount of work.
        boost::tuple<int, int, int> query_subtree_id(
          task_query_table->rank(),
          found_task.first.query_start_node()->begin(),
          found_task.first.query_start_node()->count());
        unsigned long int completed_work =
          static_cast<unsigned long int>(
            found_task.first.query_start_node()->count()) *
          static_cast<unsigned long int>(task_starting_rnode->count());
        distributed_tasks.push_completed_computation(
          query_subtree_id,
          * world_, task_starting_rnode->count(), completed_work);

        // After finishing, the lock on the query subtree is released.
        distributed_tasks.UnlockQuerySubtree(query_subtree_id);

        // Release the reference subtable.
        distributed_tasks.ReleaseCache(* world_, task_reference_cache_id, 1);

      } // end of finding a task.

      // Quit if all work is done.
      work_left_to_do = !(distributed_tasks.can_terminate());
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

  // Figure out each process's work using the global tree. Currently
  // only supports P = power of two. Fix this later.
  if(world_->size() &(world_->size() - 1)) {
    std::cerr << "Re-run with the number of processes equal to a power of "
              << "two!\n";
    return;
  }

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
