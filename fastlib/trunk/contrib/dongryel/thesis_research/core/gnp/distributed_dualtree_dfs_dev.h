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
void DistributedDualtreeDfs<DistributedProblemType>::AllToAllReduce_(
  const MetricType &metric,
  typename DistributedProblemType::ResultType *query_results) {

  // The priority queue type.
  typedef std::priority_queue <
  FrontierObjectType,
  std::vector<FrontierObjectType>,
  typename DistributedDualtreeDfs <
  DistributedProblemType >::PrioritizeTasks_ > PriorityQueueType;

  // Start the computation with the self interaction.
  DualtreeDfs<ProblemType> self_engine;
  ProblemType self_problem;
  ArgumentType self_argument;
  self_argument.Init(problem_->global());
  self_problem.Init(self_argument);
  self_engine.Init(self_problem);
  self_engine.Compute(metric, query_results);
  world_->barrier();

  // An abstract way of collaborative subtable exchanges.
  core::parallel::TableExchange <
  DistributedTableType, SubTableListType > table_exchange;
  table_exchange.Init(*world_, *reference_table_);

  // An outstanding frontier of query-reference pairs to be computed.
  std::vector < PriorityQueueType > computation_frontier(
    world_->size());
  for(unsigned int i = 0; i < computation_frontier.size(); i++) {
    if(i != static_cast<unsigned int>(world_->rank())) {
      int reference_begin = 0;
      int reference_count = reference_table_->local_n_entries(i);
      computation_frontier[i].push(
        boost::make_tuple(
          query_table_->local_table()->get_tree(),
          boost::make_tuple<int, int, int>(
            i, reference_begin, reference_count), 0.0));
    }
  }

  // The computation loop.
  bool exchange_done = false;
  do  {

    // Fill out the tasks that need to be completed in this iteration.
    std::vector< std::vector< std::pair<int, int> > > receive_requests(
      world_->size());
    PriorityQueueType prioritized_tasks;
    for(int i = 0; i < world_->size(); i++) {
      for(int j = 0; computation_frontier[i].size() > 0 &&
          j < max_num_work_to_dequeue_per_stage_; j++) {

        // Examine the top object in the frontier and sort it in the
        // priorities, while forming the request lists.
        const FrontierObjectType &top_object = computation_frontier[i].top();
        receive_requests[i].push_back(
          std::pair<int, int>(
            top_object.get<1>().get<1>(), top_object.get<1>().get<2>()));
        prioritized_tasks.push(top_object);

        // Pop the top object.
        computation_frontier[i].pop();
      }
    }

    // Try to exchange the subtables. If we are done, then we exit the
    // loop.
    if(
      exchange_done == false &&
      table_exchange.AllToAll(
        *world_, max_num_levels_to_serialize_,
        *(reference_table_->local_table()), receive_requests)) {

      // At this point, try to empty the priority queue.
      exchange_done = true;
      max_num_work_to_dequeue_per_stage_ = std::numeric_limits<int>::max();
    }
    if(exchange_done && prioritized_tasks.size() == 0) {
      break;
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
      sub_argument.Init(
        frontier_reference_subtable->table(),
        query_table_->local_table(), problem_->global());
      sub_problem.Init(sub_argument);
      sub_engine.Init(sub_problem);

      // Set the right flags and fire away the computation.
      sub_engine.set_base_case_flags(
        frontier_reference_subtable->serialize_points_per_terminal_node());
      sub_engine.set_query_reference_process_ranks(
        world_->rank(), reference_process_id);
      sub_engine.set_query_start_node(
        top_frontier.get<0>());
      sub_engine.Compute(metric, query_results, false);

      // For each unpruned query reference pair, push into the
      // priority queue.
      const std::vector < FrontierObjectType > &unpruned_query_reference_pairs =
        sub_engine.unpruned_query_reference_pairs();
      for(unsigned int k = 0;
          k < unpruned_query_reference_pairs.size(); k++) {
        computation_frontier[reference_process_id].push(
          unpruned_query_reference_pairs[k]);
      }

      // Pop the top frontier object that was just explored.
      prioritized_tasks.pop();
    } // end of taking care of the computation tasks for the current
    // iteration.
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
  max_num_levels_to_serialize_ = 15;
  max_num_work_to_dequeue_per_stage_ = 5;
}

template<typename DistributedProblemType>
void DistributedDualtreeDfs<DistributedProblemType>::set_work_params(
  int max_num_levels_to_serialize_in,
  int max_num_work_to_dequeue_per_stage_in) {

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
  PreProcessReferenceTree_(reference_table_->get_tree());
  PreProcessReferenceTree_(reference_table_->local_table()->get_tree());

  // Figure out each process's work using the global tree. a 2D matrix
  // workspace. This is currently doing an all-reduce type of
  // exchange.
  AllToAllReduce_(metric, query_results);
  world_->barrier();

  // Postprocess.
  // PostProcess_(metric, query_table_->get_tree(), query_results);
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

  typename DistributedProblemType::StatisticType &rnode_stat = rnode->stat();
  typename DistributedProblemType::DistributedTableType::TreeIterator rnode_it =
    reference_table_->get_node_iterator(rnode);

  if(rnode->is_leaf()) {
    rnode_stat.Init(rnode_it);
  }
  else {

    // Get the left and the right children.
    TemplateTreeType *rnode_left_child = rnode->left();
    TemplateTreeType *rnode_right_child = rnode->right();

    // Recurse to the left and the right.
    PreProcessReferenceTree_(rnode_left_child);
    PreProcessReferenceTree_(rnode_right_child);

    // Build the node stat by combining those owned by the children.
    typename DistributedProblemType::StatisticType &rnode_left_child_stat =
      rnode_left_child->stat();
    typename DistributedProblemType::StatisticType &rnode_right_child_stat =
      rnode_right_child->stat();
    rnode_stat.Init(
      rnode_it, rnode_left_child_stat, rnode_right_child_stat);
  }
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
