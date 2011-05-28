
/** @file distributed_dualtree_dfs_dev.h
 *
 *  The generic algorithm template for distributed dual-tree
 *  computation.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DISTRIBUTED_DUALTREE_DFS_DEV_H
#define CORE_GNP_DISTRIBUTED_DUALTREE_DFS_DEV_H

#include <boost/bind.hpp>
#include <boost/mpi.hpp>
#include <boost/tuple/tuple.hpp>
#include <map>
#include <queue>
#include "core/gnp/distributed_dualtree_dfs.h"
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/parallel/table_exchange.h"
#include "core/table/table.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace gnp {

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
DistributedProblemType >::SharedMemoryParallelize_(
  const MetricType &metric_in,
  const std::vector <
  typename DistributedDualtreeDfs<DistributedProblemType>::TreeType * >
  &local_query_subtrees,
  core::parallel::TableExchange <
  typename DistributedDualtreeDfs <
  DistributedProblemType >::DistributedTableType,
  typename DistributedDualtreeDfs <
  DistributedProblemType >::SubTableListType > &table_exchange,
  typename DistributedDualtreeDfs <
  DistributedProblemType >::CoarsePriorityQueueType &prioritized_tasks,
  typename DistributedProblemType::ResultType *query_results) {

  // The global list of query subtree that is being computed. This is
  // necessary for preventing two threads from grabbing tasks that
  // involve the same query subtree.
  std::deque<bool> active_query_subtrees(local_query_subtrees.size(), false);

  // Generate the list of tasks.
  std::vector< FinePriorityQueueType > tasks(local_query_subtrees.size());
  while(! prioritized_tasks.empty()) {

    // Examine the top object in the frontier.
    const CoarseFrontierObjectType &top_frontier = prioritized_tasks.top();

    // Find the reference process ID and grab its subtable.
    int reference_process_id = top_frontier.get<1>().get<0>();
    SubTableType *frontier_reference_subtable =
      table_exchange.FindSubTable(
        reference_process_id,
        top_frontier.get<1>().get<1>(),
        top_frontier.get<1>().get<2>());

    // Find the table and the starting reference node.
    TableType *frontier_reference_table =
      (frontier_reference_subtable != NULL) ?
      frontier_reference_subtable->table() : reference_table_->local_table();
    TreeType *reference_starting_node =
      (frontier_reference_subtable != NULL) ?
      frontier_reference_subtable->table()->get_tree() :
      reference_table_->local_table()->
      get_tree()->FindByBeginCount(
        top_frontier.get<1>().get<1>(),
        top_frontier.get<1>().get<2>());
    boost::tuple<TableType *, TreeType *> reference_table_node_pair(
      frontier_reference_table, reference_starting_node);

    // For each query subtree, create a new task.
    for(unsigned int i = 0; i < local_query_subtrees.size(); i++) {

      // Create a fine frontier object to be dequeued by each thread
      // later.
      core::math::Range squared_distance_range(
        local_query_subtrees[i]->bound().RangeDistanceSq(
          metric_in, reference_table_node_pair.get<1>()->bound()));
      FineFrontierObjectType new_task =
        boost::tuple < TreeType * ,
        boost::tuple<TableType *, TreeType *>, double > (
          local_query_subtrees[i], reference_table_node_pair,
          - squared_distance_range.mid());
      tasks[i].push(new_task);
    }

    // Make sure to pop.
    prioritized_tasks.pop();
  }

  // OpenMP parallel region. Each thread enters into a infinite loop
  // where it tries to grab an independent task.
#pragma omp parallel
  {

    do {

      std::pair<FineFrontierObjectType, int> found_task;
      found_task.second = -1;
      bool all_empty = true;
      for(unsigned int i = 0; i < tasks.size(); i++) {

#pragma omp critical
        {

          // Check whether the current query subtree is empty.
          all_empty = all_empty && (tasks[i].size() == 0);

          // Try to see if the thread can dequeue a task here.
          if((! all_empty) && (! active_query_subtrees[i]) &&
          tasks[i].size() > 0) {

            // Copy the task and the query subtree number.
            found_task.first = tasks[i].top();
            found_task.second = i;

            // Pop the task from the priority queue after copying and
            // put a lock on the query subtree.
            tasks[i].pop();
            active_query_subtrees[i] = true;
          }
        } // end of pragma omp critical

        if(found_task.second >= 0) {

          // If something is found, then break.
          break;
        }
      }

      // The thread exits if the global task list is all empty.
      if(all_empty) {
        break;
      }

      // If found something to run on, then call the serial dual-tree
      // method.
      if(found_task.second >= 0) {

        // Run a sub-dualtree algorithm on the computation object.
        core::gnp::DualtreeDfs<ProblemType> sub_engine;
        ProblemType sub_problem;
        ArgumentType sub_argument;
        TableType *task_reference_table = found_task.first.get<1>().get<0>();
        TreeType *task_starting_rnode = found_task.first.get<1>().get<1>();

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
        sub_engine.Compute(metric_in, query_results, false);

#pragma omp critical
        {
          num_deterministic_prunes_ += sub_engine.num_deterministic_prunes();
          num_probabilistic_prunes_ += sub_engine.num_probabilistic_prunes();
        }

        // After finishing, the lock on the query subtree is released.
#pragma omp critical
        {
          active_query_subtrees[ found_task.second ] = false;
        }
      } // end of finding a task.
    }
    while(true);
  } // end of pragma omp parallel.

  // Clear the cache.
  table_exchange.ClearCache();
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
    if(local_reference_node->count() <= max_reference_subtree_size) {
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
  if(local_reference_node->count() <= max_reference_subtree_size) {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node,
      essential_reference_subtrees, squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node,
      essential_reference_subtrees, squared_distance_ranges, extrinsic_prunes);
  }
  else {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->left(),
      essential_reference_subtrees, squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->right(),
      essential_reference_subtrees, squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->left(),
      essential_reference_subtrees, squared_distance_ranges, extrinsic_prunes);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->right(),
      essential_reference_subtrees, squared_distance_ranges, extrinsic_prunes);
  }
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs<DistributedProblemType>::AllToAllReduce_(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // The max number of points for the query subtree for each task.
  int max_query_subtree_size = max_subtree_size_;

  // The max number of points for the reference subtree for each task.
  int max_reference_subtree_size = max_subtree_size_;

  // For each process, break up the local query tree into a list of
  // subtree query lists.
  std::vector< TreeType *> local_query_subtrees;
  query_table_->local_table()->get_frontier_nodes(
    max_query_subtree_size, &local_query_subtrees);

  // Each process needs to customize its reference set for each
  // participating query process.
  std::vector< std::vector< std::pair<int, int> > >
  essential_reference_subtrees(world_->size());
  std::vector< std::vector< core::math::Range > >
  remote_priorities(world_->size());
  std::vector<double> extrinsic_prunes_broadcast(world_->size(), 0.0);
  ComputeEssentialReferenceSubtrees_(
    metric, max_reference_subtree_size, query_table_->get_tree(),
    reference_table_->local_table()->get_tree(), &essential_reference_subtrees,
    &remote_priorities, &extrinsic_prunes_broadcast);

  // Do an all to all to let each participating query process its
  // initial frontier.
  std::vector< std::vector< std::pair<int, int> > > reference_frontier_lists;
  std::vector< double > extrinsic_prune_lists;
  std::vector< std::vector< core::math::Range> > local_priorities;
  boost::mpi::all_to_all(
    *world_, essential_reference_subtrees, reference_frontier_lists);
  boost::mpi::all_to_all(
    *world_, remote_priorities, local_priorities);
  boost::mpi::all_to_all(
    *world_, extrinsic_prunes_broadcast, extrinsic_prune_lists);

  // Add up the initial pruned amounts and reseed it on the query
  // side.
  double initial_pruned =
    std::accumulate(
      extrinsic_prune_lists.begin(), extrinsic_prune_lists.end(), 0.0) -
    extrinsic_prune_lists[world_->rank()];
  core::gnp::DualtreeDfs<ProblemType>::PreProcess(
    query_table_->local_table(), query_table_->local_table()->get_tree(),
    query_results, initial_pruned);

  // An abstract way of collaborative subtable exchanges.
  core::parallel::TableExchange <
  DistributedTableType, SubTableListType > table_exchange;
  table_exchange.Init(
    *world_, *(reference_table_->local_table()),
    max_num_work_to_dequeue_per_stage_);

  // An outstanding frontier of query-reference pairs to be computed.
  CoarsePriorityQueueType computation_frontier;
  for(int i = 0; i < world_->size(); i++) {
    const std::vector< std::pair<int, int> > &reference_frontier =
      reference_frontier_lists[i];
    for(unsigned int j = 0; j < reference_frontier.size(); j++) {
      computation_frontier.push(
        boost::make_tuple(
          query_table_->local_table()->get_tree(),
          boost::make_tuple<int, int, int>(
            i, reference_frontier[j].first,
            reference_frontier[j].second),
          - local_priorities[i][j].mid()));
    }
  }

  printf(
    "Process %d has a total computation frontier of: %d\n",
    world_->rank(), static_cast<int>(computation_frontier.size()));

  // The computation loop.
  do  {

    // Fill out the tasks that need to be completed in this iteration.
    std::vector< std::vector< std::pair<int, int> > > receive_requests(
      world_->size());
    CoarsePriorityQueueType prioritized_tasks;
    for(int j = 0; computation_frontier.size() > 0 &&
        static_cast<int>(prioritized_tasks.size()) <
        max_num_work_to_dequeue_per_stage_; j++) {

      // Examine the top object in the frontier and sort it in the
      // priorities, while forming the request lists.
      const CoarseFrontierObjectType &top_object =
        computation_frontier.top();
      int reference_process_id = top_object.get<1>().get<0>();
      std::pair<int, int> reference_node_id(
        top_object.get<1>().get<1>(), top_object.get<1>().get<2>());

      // Each process does not need to receive anything from itself
      // (it can just do a self-local lookup).
      if(reference_process_id != world_->rank()) {
        receive_requests[reference_process_id].push_back(reference_node_id);
      }
      prioritized_tasks.push(top_object);

      // Pop the top object.
      computation_frontier.pop();
    }

    // Try to exchange the subtables.
    if(
      table_exchange.AllToAll(
        *world_, receive_requests)) {

      // Check whether all of the processes are done. Otherwise, we
      // have to be in the loop in case some processes request
      // information from me.
      bool local_done = (prioritized_tasks.size() == 0);
      bool global_done = true;
      boost::mpi::all_reduce(
        *world_, local_done, global_done, std::logical_and<bool>());
      if(global_done) {
        break;
      }
    }

    // Each process utilizes the shared memory parallelism here.
    SharedMemoryParallelize_(
      metric, local_query_subtrees, table_exchange,
      prioritized_tasks, query_results);
  }
  while(true);
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
  AllToAllReduce_(metric, query_results);
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
