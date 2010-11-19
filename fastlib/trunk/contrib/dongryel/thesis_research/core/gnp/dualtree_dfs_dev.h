/** @file dualtree_dfs_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_DEV_H
#define CORE_GNP_DUALTREE_DFS_DEV_H

#include <armadillo>
#include "dualtree_dfs.h"
#include "dualtree_dfs_iterator_dev.h"
#include "core/table/table.h"

template<typename ProblemType>
ProblemType *core::gnp::DualtreeDfs<ProblemType>::problem() {
  return problem_;
}

template<typename ProblemType>
typename ProblemType::TableType *core::gnp::DualtreeDfs<ProblemType>::query_table() {
  return query_table_;
}

template<typename ProblemType>
typename ProblemType::TableType *
core::gnp::DualtreeDfs<ProblemType>::reference_table() {
  return reference_table_;
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::ResetStatistic() {
  ResetStatisticRecursion_(query_table_->get_tree(), query_table_);
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::Init(ProblemType &problem_in) {
  problem_ = &problem_in;
  query_table_ = problem_->query_table();
  reference_table_ = problem_->reference_table();
  ResetStatistic();

  if(query_table_ != reference_table_) {
    ResetStatisticRecursion_(reference_table_->get_tree(), reference_table_);
  }
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::Compute(
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
void core::gnp::DualtreeDfs<ProblemType>::ResetStatisticRecursion_(
  typename ProblemType::TableType::TreeType *node,
  typename ProblemType::TableType * table) {
  table->get_node_stat(node).SetZero();
  if(table->node_is_leaf(node) == false) {
    ResetStatisticRecursion_(table->get_node_left_child(node), table);
    ResetStatisticRecursion_(table->get_node_right_child(node), table);
  }
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::PreProcessReferenceTree_(
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
void core::gnp::DualtreeDfs<ProblemType>::PreProcess_(
  typename ProblemType::TableType::TreeType *qnode) {

  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);
  qnode_stat.SetZero();

  if(!query_table_->node_is_leaf(qnode)) {
    PreProcess_(query_table_->get_node_left_child(qnode));
    PreProcess_(query_table_->get_node_right_child(qnode));
  }
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::DualtreeBase_(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  typename ProblemType::ResultType *query_results) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);
  qnode_stat.summary_.StartReaccumulate();

  // Postponed object to hold each query contribution.
  typename ProblemType::PostponedType query_contribution;

  // Get the query node iterator and the reference node iterator.
  typename ProblemType::TableType::TreeIterator qnode_iterator =
    query_table_->get_node_iterator(qnode);
  typename ProblemType::TableType::TreeIterator rnode_iterator =
    reference_table_->get_node_iterator(rnode);

  // Compute unnormalized sum for each query point.
  while(qnode_iterator.HasNext()) {

    // Get the query point and its real index.
    core::table::DenseConstPoint q_col;
    int q_index;
    qnode_iterator.Next(&q_col, &q_index);

    // Reset the temporary variable for accumulating each
    // reference point contribution.
    query_contribution.Init(reference_table_->get_node_count(rnode));

    // Incorporate the postponed information.
    query_results->ApplyPostponed(q_index, qnode_stat.postponed_);

    // Reset the reference node iterator.
    rnode_iterator.Reset();
    while(rnode_iterator.HasNext()) {

      // Get the reference point and accumulate contribution.
      core::table::DenseConstPoint r_col;
      int r_col_id;
      rnode_iterator.Next(&r_col, &r_col_id);
      query_contribution.ApplyContribution(
        problem_->global(), metric, q_col, r_col);

    } // end of iterating over each reference point.

    // Each query point has taken care of all reference points.
    query_results->ApplyPostponed(q_index, query_contribution);

    // Refine min and max summary statistics.
    qnode_stat.summary_.Accumulate(
      problem_->global(), *query_results, q_index);

  } // end of looping over each query point.

  // Clear postponed information.
  qnode_stat.postponed_.SetZero();
}

template<typename ProblemType>
bool core::gnp::DualtreeDfs<ProblemType>::CanProbabilisticSummarize_(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  double failure_probability,
  typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);
  typename ProblemType::SummaryType new_summary(qnode_stat.summary_);
  new_summary.ApplyPostponed(qnode_stat.postponed_);
  new_summary.ApplyDelta(delta);

  return new_summary.CanProbabilisticSummarize(
           metric, problem_->global(), delta, qnode, rnode,
           failure_probability, query_results);
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::ProbabilisticSummarize_(
  typename ProblemType::GlobalType &global,
  typename ProblemType::TableType::TreeType *qnode,
  double failure_probability,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  query_results->ApplyProbabilisticDelta(
    global, qnode, failure_probability, delta);
}

template<typename ProblemType>
bool core::gnp::DualtreeDfs<ProblemType>::CanSummarize_(
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);
  typename ProblemType::SummaryType new_summary(qnode_stat.summary_);
  new_summary.ApplyPostponed(qnode_stat.postponed_);
  new_summary.ApplyDelta(delta);

  return new_summary.CanSummarize(problem_->global(), delta, qnode, rnode,
                                  query_results);
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::Summarize_(
  typename ProblemType::TableType::TreeType *qnode,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);
  qnode_stat.postponed_.ApplyDelta(delta, query_results);
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::Heuristic_(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::TableType::TreeType *node,
  typename ProblemType::TableType *node_table,
  typename ProblemType::TableType::TreeType *first_candidate,
  typename ProblemType::TableType::TreeType *second_candidate,
  typename ProblemType::TableType *candidate_table,
  typename ProblemType::TableType::TreeType **first_partner,
  core::math::Range &first_squared_distance_range,
  typename ProblemType::TableType::TreeType **second_partner,
  core::math::Range &second_squared_distance_range) {

  core::math::Range tmp_first_squared_distance_range =
    node_table->get_node_bound(node).RangeDistanceSq(
      metric,
      candidate_table->get_node_bound(first_candidate));
  core::math::Range tmp_second_squared_distance_range =
    node_table->get_node_bound(node).RangeDistanceSq(
      metric,
      candidate_table->get_node_bound(second_candidate));

  if(tmp_first_squared_distance_range.lo <=
      tmp_second_squared_distance_range.lo) {
    *first_partner = first_candidate;
    first_squared_distance_range = tmp_first_squared_distance_range;
    *second_partner = second_candidate;
    second_squared_distance_range = tmp_second_squared_distance_range;
  }
  else {
    *first_partner = second_candidate;
    first_squared_distance_range = tmp_second_squared_distance_range;
    *second_partner = first_candidate;
    second_squared_distance_range = tmp_first_squared_distance_range;
  }
}

template<typename ProblemType>
bool core::gnp::DualtreeDfs<ProblemType>::DualtreeCanonical_(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  double failure_probability,
  const core::math::Range &squared_distance_range,
  typename ProblemType::ResultType *query_results) {

  // Compute the delta change.
  typename ProblemType::DeltaType delta;
  delta.DeterministicCompute(metric, problem_->global(), qnode, rnode,
                             squared_distance_range);

  // If it is prunable, then summarize and return.
  if(CanSummarize_(qnode, rnode, delta, query_results)) {
    Summarize_(qnode, delta, query_results);
    return true;
  }
  else if(failure_probability > 1e-6) {

    // Try Monte Carlo.
    if(CanProbabilisticSummarize_(metric, qnode, rnode,
                                  failure_probability,
                                  delta, query_results)) {
      ProbabilisticSummarize_(problem_->global(), qnode,
                              failure_probability,
                              delta, query_results);
      return false;
    }
  }

  // If it is not prunable and the query node is a leaf,
  if(query_table_->node_is_leaf(qnode)) {

    bool exact_compute = true;
    if(reference_table_->node_is_leaf(rnode)) {
      DualtreeBase_(metric, qnode, rnode, query_results);
    } // qnode is leaf, rnode is leaf.
    else {
      typename ProblemType::TableType::TreeType *rnode_first;
      core::math::Range squared_distance_range_first,
           squared_distance_range_second;
      typename ProblemType::TableType::TreeType *rnode_second;
      Heuristic_(metric, qnode, query_table_,
                 reference_table_->get_node_left_child(rnode),
                 reference_table_->get_node_right_child(rnode),
                 reference_table_,
                 &rnode_first, squared_distance_range_first,
                 &rnode_second, squared_distance_range_second);

      // Recurse.
      bool rnode_first_exact =
        DualtreeCanonical_(metric,
                           qnode,
                           rnode_first,
                           failure_probability / 2.0,
                           squared_distance_range_first,
                           query_results);

      bool rnode_second_exact =
        DualtreeCanonical_(metric,
                           qnode,
                           rnode_second,
                           (rnode_first_exact) ?
                           failure_probability : failure_probability / 2.0,
                           squared_distance_range_second,
                           query_results);
      exact_compute = rnode_first_exact && rnode_second_exact;
    } // qnode is leaf, rnode is not leaf.
    return exact_compute;
  } // end of query node being a leaf.

  // If we are here, we have to split the query.
  bool exact_compute_nonleaf_qnode = true;

  // Get the current query node statistic.
  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);

  // Left and right nodes of the query node and their statistic.
  typename ProblemType::TableType::TreeType *qnode_left =
    query_table_->get_node_left_child(qnode);
  typename ProblemType::TableType::TreeType *qnode_right =
    query_table_->get_node_right_child(qnode);
  typename ProblemType::StatisticType &qnode_left_stat =
    query_table_->get_node_stat(qnode_left);
  typename ProblemType::StatisticType &qnode_right_stat =
    query_table_->get_node_stat(qnode_right);

  // Push down postponed and clear.
  qnode_left_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
  qnode_right_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
  qnode_stat.postponed_.SetZero();

  if(reference_table_->node_is_leaf(rnode)) {
    typename ProblemType::TableType::TreeType *qnode_first;
    core::math::Range
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::TableType::TreeType *qnode_second;
    Heuristic_(metric, rnode, reference_table_, qnode_left, qnode_right,
               query_table_, &qnode_first, squared_distance_range_first,
               &qnode_second, squared_distance_range_second);

    // Recurse.
    bool first_qnode_exact = DualtreeCanonical_(metric,
                             qnode_first,
                             rnode,
                             failure_probability,
                             squared_distance_range_first,
                             query_results);
    bool second_qnode_exact = DualtreeCanonical_(metric,
                              qnode_second,
                              rnode,
                              failure_probability,
                              squared_distance_range_second,
                              query_results);
    exact_compute_nonleaf_qnode = first_qnode_exact && second_qnode_exact;
  } // qnode is not leaf, rnode is leaf.

  else {
    typename ProblemType::TableType::TreeType *rnode_first;
    core::math::Range
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::TableType::TreeType *rnode_second;
    Heuristic_(
      metric,
      qnode_left,
      query_table_,
      reference_table_->get_node_left_child(rnode),
      reference_table_->get_node_right_child(rnode),
      reference_table_,
      &rnode_first,
      squared_distance_range_first,
      &rnode_second, squared_distance_range_second);

    // Recurse.
    bool qnode_left_rnode_first_exact = DualtreeCanonical_(
                                          metric,
                                          qnode_left,
                                          rnode_first,
                                          failure_probability / 2.0,
                                          squared_distance_range_first,
                                          query_results);
    bool qnode_left_rnode_second_exact = DualtreeCanonical_(
                                           metric,
                                           qnode_left,
                                           rnode_second,
                                           (qnode_left_rnode_first_exact) ?
                                           failure_probability : failure_probability / 2.0,
                                           squared_distance_range_second,
                                           query_results);

    Heuristic_(
      metric,
      qnode_right,
      query_table_,
      reference_table_->get_node_left_child(rnode),
      reference_table_->get_node_right_child(rnode),
      reference_table_,
      &rnode_first,
      squared_distance_range_first,
      &rnode_second,
      squared_distance_range_second);

    // Recurse.
    bool qnode_right_rnode_first_exact = DualtreeCanonical_(
                                           metric,
                                           qnode_right,
                                           rnode_first,
                                           failure_probability / 2.0,
                                           squared_distance_range_first,
                                           query_results);
    bool qnode_right_rnode_second_exact = DualtreeCanonical_(
                                            metric,
                                            qnode_right,
                                            rnode_second,
                                            (qnode_right_rnode_first_exact) ?
                                            failure_probability : failure_probability / 2.0,
                                            squared_distance_range_second,
                                            query_results);

    // Merge the boolean results.
    exact_compute_nonleaf_qnode = qnode_left_rnode_first_exact &&
                                  qnode_left_rnode_second_exact &&
                                  qnode_right_rnode_first_exact &&
                                  qnode_right_rnode_second_exact;

  } // qnode is not leaf, rnode is not leaf.

  // Reset summary results of the current query node.
  qnode_stat.summary_.StartReaccumulate();
  qnode_stat.summary_.Accumulate(
    problem_->global(), qnode_left_stat.summary_, qnode_left_stat.postponed_);
  qnode_stat.summary_.Accumulate(
    problem_->global(), qnode_right_stat.summary_, qnode_right_stat.postponed_);

  return exact_compute_nonleaf_qnode;
}

template<typename ProblemType>
void core::gnp::DualtreeDfs<ProblemType>::PostProcess_(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat =
    query_table_->get_node_stat(qnode);

  if(query_table_->node_is_leaf(qnode)) {

    typename ProblemType::TableType::TreeIterator qnode_iterator =
      query_table_->get_node_iterator(qnode);

    while(qnode_iterator.HasNext()) {
      core::table::DenseConstPoint q_col;
      int q_index;
      qnode_iterator.Next(&q_col, &q_index);
      query_results->ApplyPostponed(q_index, qnode_stat.postponed_);
      query_results->PostProcess(metric, q_index, problem_->global(),
                                 problem_->is_monochromatic());
    }
    qnode_stat.postponed_.SetZero();
  }
  else {
    typename ProblemType::TableType::TreeType *qnode_left =
      query_table_->get_node_left_child(qnode);
    typename ProblemType::TableType::TreeType *qnode_right =
      query_table_->get_node_right_child(qnode);
    typename ProblemType::StatisticType &qnode_left_stat =
      query_table_->get_node_stat(qnode_left);
    typename ProblemType::StatisticType &qnode_right_stat =
      query_table_->get_node_stat(qnode_right);

    qnode_left_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
    qnode_right_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
    qnode_stat.postponed_.SetZero();

    PostProcess_(metric, qnode_left,  query_results);
    PostProcess_(metric, qnode_right, query_results);
  }
}

#endif
