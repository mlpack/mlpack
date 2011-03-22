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

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs <
DistributedProblemType >::ComputeEssentialReferenceSubtrees_(
  const MetricType &metric_in,
  int max_reference_subtree_size,
  DistributedTreeType *global_query_node, TreeType *local_reference_node,
  std::vector <
  std::vector< std::pair<int, int> > > *essential_reference_subtrees) {

  // Compute the squared distance ranges between the query node and
  // the reference node.
  core::math::Range squared_distance_range =
    global_query_node->bound().RangeDistanceSq(
      metric_in, local_reference_node->bound());

  // If the pair is prunable, then return.
  if(problem_->global().ConsiderExtrinsicPrune(squared_distance_range)) {
    printf("Pruned %d %d %d %d\n", global_query_node->begin(),
           global_query_node->count(),
           local_reference_node->begin(), local_reference_node->end());
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
      }
    }
    else {

      ComputeEssentialReferenceSubtrees_(
        metric_in, max_reference_subtree_size,
        global_query_node, local_reference_node->left(),
        essential_reference_subtrees);
      ComputeEssentialReferenceSubtrees_(
        metric_in, max_reference_subtree_size,
        global_query_node, local_reference_node->right(),
        essential_reference_subtrees);
    }
    return;
  }

  // Here, we know that the global query node is a non-leaf, so we
  // need to split both ways.
  if(local_reference_node->count() <= max_reference_subtree_size) {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node,
      essential_reference_subtrees);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node,
      essential_reference_subtrees);
  }
  else {
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->left(),
      essential_reference_subtrees);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->left(), local_reference_node->right(),
      essential_reference_subtrees);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->left(),
      essential_reference_subtrees);
    ComputeEssentialReferenceSubtrees_(
      metric_in, max_reference_subtree_size,
      global_query_node->right(), local_reference_node->right(),
      essential_reference_subtrees);
  }
}

template<typename DistributedProblemType>
template<typename MetricType>
void DistributedDualtreeDfs<DistributedProblemType>::AllToAllReduce_(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // Each process needs to customize its reference set for each
  // participating query process.
  const int max_reference_subtree_size = 20000;
  std::vector< std::vector< std::pair<int, int> > >
  essential_reference_subtrees(world_->size());
  ComputeEssentialReferenceSubtrees_(
    metric, max_reference_subtree_size, query_table_->get_tree(),
    reference_table_->local_table()->get_tree(), &essential_reference_subtrees);

  // Do an all to all to let each participating query process its
  // initial frontier.
  std::vector< std::vector< std::pair<int, int> > > reference_frontier_lists;
  boost::mpi::all_to_all(
    *world_, essential_reference_subtrees, reference_frontier_lists);

  // The priority queue type.
  typedef std::priority_queue <
  FrontierObjectType,
  std::vector<FrontierObjectType>,
  typename DistributedDualtreeDfs <
  DistributedProblemType >::PrioritizeTasks_ > PriorityQueueType;

  // An abstract way of collaborative subtable exchanges.
  core::parallel::TableExchange <
  DistributedTableType, SubTableListType > table_exchange;
  table_exchange.Init(
    *world_, *(reference_table_->local_table()),
    max_num_work_to_dequeue_per_stage_);

  // An outstanding frontier of query-reference pairs to be computed.
  std::vector < PriorityQueueType > computation_frontier(
    world_->size());
  for(unsigned int i = 0; i < computation_frontier.size(); i++) {
    const std::vector< std::pair<int, int> > &reference_frontier =
      reference_frontier_lists[i];
    for(unsigned int j = 0; j < reference_frontier.size(); j++) {
      computation_frontier[i].push(
        boost::make_tuple(
          query_table_->local_table()->get_tree(),
          boost::make_tuple<int, int, int>(
            i, reference_frontier[j].first,
            reference_frontier[j].second), 0.0));
      printf("Process %d: %d %d %d\n", world_->rank(),
             i, reference_frontier[j].first,
             reference_frontier[j].second);
    }
  }

  // The computation loop.
  do  {

    // Fill out the tasks that need to be completed in this iteration.
    std::vector< std::vector< std::pair<int, int> > > receive_requests(
      world_->size());
    PriorityQueueType prioritized_tasks;
    int current_computation_frontier_size = 0;
    for(int i = 0; i < world_->size(); i++) {
      current_computation_frontier_size += computation_frontier[i].size();
      for(int j = 0; computation_frontier[i].size() > 0 &&
          j < max_num_work_to_dequeue_per_stage_; j++) {

        // Examine the top object in the frontier and sort it in the
        // priorities, while forming the request lists.
        const FrontierObjectType &top_object = computation_frontier[i].top();
        std::pair<int, int> reference_node_id(
          top_object.get<1>().get<1>(), top_object.get<1>().get<2>());

        // Each process does not need to receive anything from itself
        // (it can just do a self-local lookup).
        if(table_exchange.FindSubTable(
              i, reference_node_id.first,
              reference_node_id.second) == NULL && i != world_->rank() &&
            std::find(
              receive_requests[i].begin(), receive_requests[i].end(),
              reference_node_id) == receive_requests[i].end()) {
          receive_requests[i].push_back(reference_node_id);
        }
        prioritized_tasks.push(top_object);

        // Pop the top object.
        computation_frontier[i].pop();
      }
    }

    // Update the computation frontier size statistics.
    max_computation_frontier_size_ =
      std::max(
        max_computation_frontier_size_, current_computation_frontier_size);

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

    // Each process calls the independent sets of serial dual-tree dfs
    // algorithms. Further parallelism can be exploited here.
    while(! prioritized_tasks.empty()) {

      // Examine the top object in the frontier.
      const FrontierObjectType &top_frontier = prioritized_tasks.top();

      // Run a sub-dualtree algorithm on the computation object.
      core::gnp::DualtreeDfs<ProblemType> sub_engine;
      ProblemType sub_problem;
      ArgumentType sub_argument;
      int reference_process_id = top_frontier.get<1>().get<0>();
      SubTableType *frontier_reference_subtable =
        table_exchange.FindSubTable(
          reference_process_id,
          top_frontier.get<1>().get<1>(),
          top_frontier.get<1>().get<2>());
      printf("Process %d takes care of %d %d %d\n", world_->rank(),
             reference_process_id,
             top_frontier.get<1>().get<1>(),
             top_frontier.get<1>().get<2>());
      sub_argument.Init(
        (frontier_reference_subtable) ?
        frontier_reference_subtable->table() : reference_table_->local_table(),
        query_table_->local_table(), problem_->global());
      sub_problem.Init(sub_argument, &(problem_->global()));
      sub_engine.Init(sub_problem);

      // Set the right flags and fire away the computation.
      if(frontier_reference_subtable != NULL) {
        sub_engine.set_base_case_flags(
          frontier_reference_subtable->serialize_points_per_terminal_node());
      }
      sub_engine.set_query_reference_process_ranks(
        world_->rank(), reference_process_id);
      if(frontier_reference_subtable == NULL) {
        TreeType *reference_start_node =
          reference_table_->local_table()->
          get_tree()->FindByBeginCount(
            top_frontier.get<1>().get<1>(),
            top_frontier.get<1>().get<2>());
        sub_engine.set_reference_start_node(reference_start_node);
      }
      sub_engine.Compute(metric, query_results, false);

      // Pop the top frontier object that was just explored.
      prioritized_tasks.pop();

    } // end of taking care of the computation tasks for the current
    // iteration.
  }
  while(true);

  printf("The maximum computation frontier size during the computation: %d "
         "for Process %d\n", max_computation_frontier_size_, world_->rank());
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
  max_num_levels_to_serialize_ = 15;
  max_num_work_to_dequeue_per_stage_ = 5;
  max_computation_frontier_size_ = 0;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs<DistributedProblemType>::set_work_params(
  int leaf_size_in,
  int max_num_levels_to_serialize_in,
  int max_num_work_to_dequeue_per_stage_in) {

  leaf_size_ = leaf_size_in;
  max_num_levels_to_serialize_ = max_num_levels_to_serialize_in;
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
  query_results->Init(query_table_->n_entries());

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
