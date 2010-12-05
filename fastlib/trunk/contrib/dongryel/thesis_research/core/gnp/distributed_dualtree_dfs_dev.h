/** @file distributed_dualtree_dfs_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DISTRIBUTED_DUALTREE_DFS_DEV_H
#define CORE_GNP_DISTRIBUTED_DUALTREE_DFS_DEV_H

#include "distributed_dualtree_dfs.h"
#include "core/table/table.h"

template<typename ProblemType>
ProblemType *core::gnp::DistributedDualtreeDfs<ProblemType>::problem() {
  return problem_;
}

template<typename ProblemType>
typename ProblemType::DistributedTableType *
core::gnp::DistributedDualtreeDfs<ProblemType>::query_table() {
  return query_table_;
}

template<typename ProblemType>
typename ProblemType::DistributedTableType *
core::gnp::DistributedDualtreeDfs<ProblemType>::reference_table() {
  return reference_table_;
}

template<typename ProblemType>
void core::gnp::DistributedDualtreeDfs<ProblemType>::ResetStatistic() {
  ResetStatisticRecursion_(query_table_->get_tree(), query_table_);
}

template<typename ProblemType>
void core::gnp::DistributedDualtreeDfs<ProblemType>::Init(ProblemType &problem_in) {
  problem_ = &problem_in;
  query_table_ = problem_->query_table();
  reference_table_ = problem_->reference_table();
  ResetStatistic();

  if(query_table_ != reference_table_) {
    ResetStatisticRecursion_(reference_table_->get_tree(), reference_table_);
  }
}

template<typename ProblemType>
void core::gnp::DistributedDualtreeDfs<ProblemType>::Compute(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::ResultType *query_results) {

  // Allocate space for storing the final results.
  query_results->Init(query_table_->n_entries());

  // Call the algorithm computation.
  core::math::Range squared_distance_range =
    (query_table_->get_node_bound(query_table_->get_tree())).RangeDistanceSq(
      metric,
      reference_table_->get_node_bound
      (reference_table_->get_tree()));

  PreProcess_(query_table_->get_tree());
  PreProcessReferenceTree_(reference_table_->get_tree());
  DualtreeCanonical_(metric,
                     query_table_->get_tree(),
                     reference_table_->get_tree(),
                     1.0 - problem_->global().probability(),
                     squared_distance_range,
                     query_results);
  PostProcess_(metric, query_table_->get_tree(), query_results);
}

template<typename ProblemType>
void core::gnp::DistributedDualtreeDfs<ProblemType>::ResetStatisticRecursion_(
  typename ProblemType::TableType::TreeType *node,
  typename ProblemType::TableType * table) {
  table->get_node_stat(node).SetZero();
  if(table->node_is_leaf(node) == false) {
    ResetStatisticRecursion_(table->get_node_left_child(node), table);
    ResetStatisticRecursion_(table->get_node_right_child(node), table);
  }
}

template<typename ProblemType>
void core::gnp::DistributedDualtreeDfs<ProblemType>::PreProcessReferenceTree_(
  typename ProblemType::TableType::TreeType *rnode) {

  typename ProblemType::StatisticType &rnode_stat =
    reference_table_->get_node_stat(rnode);
  typename ProblemType::TableType::TreeIterator rnode_it =
    reference_table_->get_node_iterator(rnode);

  if(reference_table_->node_is_leaf(rnode)) {
    rnode_stat.Init(rnode_it);
  }
  else {

    // Get the left and the right children.
    typename ProblemType::TableType::TreeType *rnode_left_child =
      reference_table_->get_node_left_child(rnode);
    typename ProblemType::TableType::TreeType *rnode_right_child =
      reference_table_->get_node_right_child(rnode);

    // Recurse to the left and the right.
    PreProcessReferenceTree_(rnode_left_child);
    PreProcessReferenceTree_(rnode_right_child);

    // Build the node stat by combining those owned by the children.
    typename ProblemType::StatisticType &rnode_left_child_stat =
      reference_table_->get_node_stat(rnode_left_child) ;
    typename ProblemType::StatisticType &rnode_right_child_stat =
      reference_table_->get_node_stat(rnode_right_child) ;
    rnode_stat.Init(
      rnode_it, rnode_left_child_stat, rnode_right_child_stat);
  }
}

template<typename ProblemType>
void core::gnp::DistributedDualtreeDfs<ProblemType>::PreProcess_(
  typename ProblemType::TableType::TreeType *qnode) {

  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);
  qnode_stat.SetZero();

  if(!query_table_->node_is_leaf(qnode)) {
    PreProcess_(query_table_->get_node_left_child(qnode));
    PreProcess_(query_table_->get_node_right_child(qnode));
  }
}

#endif
