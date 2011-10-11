/** @file dualtree_dfs_dev.h
 *
 *  An implementation of the template generator for dualtree problems.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_DEV_H
#define CORE_GNP_DUALTREE_DFS_DEV_H

#include <armadillo>
#include "core/gnp/dualtree_dfs.h"
#include "core/gnp/dualtree_dfs_iterator_dev.h"
#include "core/table/table.h"

namespace core {
namespace gnp {

template<typename ProblemType>
void DualtreeDfs<ProblemType>::set_query_start_node(
  TreeType *query_start_node_in) {
  query_start_node_ = query_start_node_in;
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::set_reference_start_node(
  TreeType *reference_start_node_in) {
  reference_start_node_ = reference_start_node_in;
}

template<typename ProblemType>
DualtreeDfs<ProblemType>::DualtreeDfs() {
  query_start_node_ = NULL;
  reference_start_node_ = NULL;
  num_deterministic_prunes_ = 0;
  num_probabilistic_prunes_ = 0;
}

template<typename ProblemType>
int DualtreeDfs<ProblemType>::num_deterministic_prunes() const {
  return num_deterministic_prunes_;
}

template<typename ProblemType>
int DualtreeDfs<ProblemType>::num_probabilistic_prunes() const {
  return num_probabilistic_prunes_;
}

template<typename ProblemType>
ProblemType *DualtreeDfs<ProblemType>::problem() {
  return problem_;
}

template<typename ProblemType>
typename ProblemType::TableType *DualtreeDfs<ProblemType>::query_table() {
  return query_table_;
}

template<typename ProblemType>
typename ProblemType::TableType *
DualtreeDfs<ProblemType>::reference_table() {
  return reference_table_;
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::Init(ProblemType &problem_in) {

  // Reset prune statistics.
  num_deterministic_prunes_ = 0;
  num_probabilistic_prunes_ = 0;

  // Set the problem.
  problem_ = &problem_in;

  // Set the query table.
  query_table_ = problem_->query_table();
  query_start_node_ = query_table_->get_tree();

  // Set the reference table and its starting node.
  reference_table_ = problem_->reference_table();
  reference_start_node_ = reference_table_->get_tree();
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::Compute(
  const MetricType &metric,
  typename ProblemType::ResultType *query_results,
  bool do_initializations) {

  // Allocate space for storing the final results.
  if(do_initializations) {
    query_results->Init(
      problem_->global(), query_table_->n_entries());
  }

  // Call the algorithm computation.
  core::math::Range squared_distance_range =
    (query_start_node_->bound()).RangeDistanceSq(
      metric, reference_start_node_->bound());

  if(do_initializations) {

    // Initialze the reference tree.
    core::gnp::DualtreeDfs <
    ProblemType >::PreProcessReferenceTree(
      problem_->global(), reference_start_node_);

    // Then the query tree.
    PreProcess(query_table_, query_table_->get_tree(), query_results, 0.0);
  }

  if(problem_->global().is_monochromatic() &&
      query_table_->rank() == reference_table_->rank()) {

    // Take care such that the monochromatic case is handled
    // correctly.
    if(query_start_node_->count() > reference_start_node_->count() &&
        query_start_node_->begin() <= reference_start_node_->begin() &&
        reference_start_node_->end() <= query_start_node_->end()) {

      std::vector<TreeType *> query_start_node_sublist;
      query_start_node_->get_frontier_nodes_disjoint_from(
        reference_start_node_, &query_start_node_sublist);

      for(unsigned int i = 0; i < query_start_node_sublist.size(); i++) {
        core::math::Range sub_squared_distance_range =
          (query_start_node_sublist[i]->bound()).RangeDistanceSq(
            metric, reference_start_node_->bound());
        DualtreeCanonical_(
          metric, query_start_node_sublist[i], reference_start_node_,
          1.0 - problem_->global().probability(), sub_squared_distance_range,
          query_results);
      }
    }
    else if(reference_start_node_->count() > query_start_node_->count() &&
            reference_start_node_->begin() <= query_start_node_->begin() &&
            query_start_node_->end() <= reference_start_node_->end()) {

      std::vector<TreeType *> reference_start_node_sublist;
      reference_start_node_->get_frontier_nodes_disjoint_from(
        query_start_node_, &reference_start_node_sublist);

      for(unsigned int i = 0; i < reference_start_node_sublist.size(); i++) {
        core::math::Range sub_squared_distance_range =
          (query_start_node_->bound()).RangeDistanceSq(
            metric, reference_start_node_sublist[i]->bound());
        DualtreeCanonical_(
          metric, query_start_node_, reference_start_node_sublist[i],
          1.0 - problem_->global().probability(), sub_squared_distance_range,
          query_results);
      }
    }
    else {
      DualtreeCanonical_(
        metric, query_start_node_, reference_start_node_,
        1.0 - problem_->global().probability(), squared_distance_range,
        query_results);
    }
  }
  else {
    DualtreeCanonical_(
      metric, query_start_node_, reference_start_node_,
      1.0 - problem_->global().probability(), squared_distance_range,
      query_results);
  }

  // Postprocess.
  PostProcess_(metric, query_start_node_, query_results, true);
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::PreProcessReferenceTree(
  typename ProblemType::GlobalType &global_in,
  typename ProblemType::TableType::TreeType *rnode) {

  typename ProblemType::StatisticType &rnode_stat = rnode->stat();

  if(rnode->is_leaf()) {
    rnode_stat.Init(global_in, rnode);
  }
  else {

    // Get the left and the right children.
    typename ProblemType::TableType::TreeType *rnode_left_child = rnode->left();
    typename ProblemType::TableType::TreeType *rnode_right_child =
      rnode->right();

    // Recurse to the left and the right.
    PreProcessReferenceTree(global_in, rnode_left_child);
    PreProcessReferenceTree(global_in, rnode_right_child);

    // Build the node stat by combining those owned by the children.
    typename ProblemType::StatisticType &rnode_left_child_stat =
      rnode_left_child->stat();
    typename ProblemType::StatisticType &rnode_right_child_stat =
      rnode_right_child->stat();
    rnode_stat.Init(
      global_in, rnode, rnode_left_child_stat, rnode_right_child_stat);
  }
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::PreProcess(
  TableType *query_table_in,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::ResultType *query_results,
  double initial_pruned) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.Seed(initial_pruned);

  if(! qnode->is_leaf()) {
    PreProcess(query_table_in, qnode->left(), query_results, initial_pruned);
    PreProcess(query_table_in, qnode->right(), query_results, initial_pruned);
  }
  else {
    typename TableType::TreeIterator qnode_it =
      query_table_in->get_node_iterator(qnode);
    while(qnode_it.HasNext()) {
      int qpoint_index;
      qnode_it.Next(&qpoint_index);
      query_results->Seed(qpoint_index, initial_pruned);
    }
  }
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::DualtreeBase_(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  bool qnode_and_rnode_are_equal,
  typename ProblemType::ResultType *query_results) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.summary_.StartReaccumulate();

  // Postponed object to hold each query contribution.
  typename ProblemType::PostponedType query_contribution;
  query_contribution.Init(problem_->global());

  // Get the query node iterator and the reference node iterator.
  typename ProblemType::TableType::TreeIterator qnode_iterator =
    query_table_->get_node_iterator(qnode);
  typename ProblemType::TableType::TreeIterator rnode_iterator =
    reference_table_->get_node_iterator(rnode);

  // Compute unnormalized sum for each query point.
  int q_dfs_index = qnode->begin();
  while(qnode_iterator.HasNext()) {

    // Get the query point and its real index.
    arma::vec q_col;
    int q_col_id;
    double q_weight;
    qnode_iterator.Next(&q_col, &q_col_id, &q_weight);

    // Reset the temporary variable for accumulating each
    // reference point contribution.
    query_contribution.Init(
      problem_->global(), qnode, rnode, qnode_and_rnode_are_equal);

    // Incorporate the postponed information.
    query_results->ApplyPostponed(q_col_id, qnode_stat.postponed_);

    // Reset the reference node iterator.
    rnode_iterator.Reset();
    int r_dfs_index = rnode->begin();
    while(rnode_iterator.HasNext()) {

      // Get the reference point and accumulate contribution.
      arma::vec r_col;
      int r_col_id;
      double r_weight;
      rnode_iterator.Next(&r_col, &r_col_id, &r_weight);
      query_contribution.ApplyContribution(
        problem_->global(), metric,
        q_col, query_table_->rank(), q_dfs_index, q_weight,
        r_col, reference_table_->rank(), r_dfs_index, r_weight);
      r_dfs_index++;

    } // end of iterating over each reference point.

    // Each query point has taken care of all reference points.
    query_results->ApplyPostponed(q_col_id, query_contribution);

    // Refine min and max summary statistics.
    qnode_stat.summary_.Accumulate(
      problem_->global(), *query_results, q_col_id);
    q_dfs_index++;

  } // end of looping over each query point.

  // Clear postponed information.
  qnode_stat.postponed_.SetZero();
}

template<typename ProblemType>
template<typename MetricType>
bool DualtreeDfs<ProblemType>::CanProbabilisticSummarize_(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  bool qnode_and_rnode_are_equal,
  double failure_probability,
  typename ProblemType::DeltaType &delta,
  const core::math::Range &squared_distance_range,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  typename ProblemType::SummaryType new_summary(qnode_stat.summary_);
  new_summary.ApplyPostponed(qnode_stat.postponed_);
  new_summary.ApplyDelta(delta);

  return new_summary.CanProbabilisticSummarize(
           metric, problem_->global(), qnode_stat.postponed_, delta,
           squared_distance_range, qnode, query_table_->rank(),
           rnode, reference_table_->rank(), qnode_and_rnode_are_equal,
           failure_probability, query_results);
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::ProbabilisticSummarize_(
  const MetricType &metric,
  typename ProblemType::GlobalType &global,
  typename ProblemType::TableType::TreeType *qnode,
  double failure_probability,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  query_results->ApplyProbabilisticDelta(
    global, qnode, failure_probability, delta);
  PostProcess_(metric, qnode, query_results, false);
}

template<typename ProblemType>
bool DualtreeDfs<ProblemType>::CanSummarize_(
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  bool qnode_and_rnode_are_equal,
  typename ProblemType::DeltaType &delta,
  const core::math::Range &squared_distance_range,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  typename ProblemType::SummaryType new_summary(qnode_stat.summary_);
  new_summary.ApplyPostponed(qnode_stat.postponed_);
  new_summary.ApplyDelta(delta);

  return new_summary.CanSummarize(
           problem_->global(), delta, squared_distance_range,
           qnode, query_table_->rank(), rnode, reference_table_->rank(),
           qnode_and_rnode_are_equal, query_results);
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::Summarize_(
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.postponed_.ApplyDelta(
    qnode, rnode, problem_->global(), delta, query_results);
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::Heuristic(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *node,
  typename ProblemType::TableType::TreeType *first_candidate,
  typename ProblemType::TableType::TreeType *second_candidate,
  typename ProblemType::TableType::TreeType **first_partner,
  core::math::Range &first_squared_distance_range,
  typename ProblemType::TableType::TreeType **second_partner,
  core::math::Range &second_squared_distance_range) {

  core::math::Range tmp_first_squared_distance_range =
    (node->bound()).RangeDistanceSq(
      metric, first_candidate->bound());
  core::math::Range tmp_second_squared_distance_range =
    (node->bound()).RangeDistanceSq(
      metric, second_candidate->bound());

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
template<typename MetricType>
bool DualtreeDfs<ProblemType>::DualtreeCanonical_(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  double failure_probability,
  const core::math::Range &squared_distance_range,
  typename ProblemType::ResultType *query_results) {

  // Flag whether qnode and rnode are equal.
  bool qnode_and_rnode_are_equal =
    (query_table_->rank() == reference_table_->rank() &&
     qnode->begin() == rnode->begin() &&
     qnode->count() == rnode->count());

  // Compute the delta change.
  typename ProblemType::DeltaType delta;
  delta.DeterministicCompute(
    metric, problem_->global(), qnode, rnode,
    qnode_and_rnode_are_equal,
    squared_distance_range);

  // If it is prunable, then summarize and return.
  if(CanSummarize_(
        qnode, rnode, qnode_and_rnode_are_equal,
        delta, squared_distance_range, query_results)) {
    Summarize_(qnode, rnode, delta, query_results);
    num_deterministic_prunes_++;
    return true;
  }
  else if(failure_probability > 1e-6) {

    // Try Monte Carlo.
    if(CanProbabilisticSummarize_(
          metric, qnode, rnode, qnode_and_rnode_are_equal, failure_probability,
          delta, squared_distance_range, query_results)) {
      ProbabilisticSummarize_(
        metric, problem_->global(), qnode,
        failure_probability, delta, query_results);
      num_probabilistic_prunes_++;
      return false;
    }
  }

  // If it is not prunable and the query node is a leaf,
  if(qnode->is_leaf()) {

    bool exact_compute = true;
    if(rnode->is_leaf()) {

      // If the base case must be done, then do so.
      DualtreeBase_(
        metric, qnode, rnode, qnode_and_rnode_are_equal, query_results);

    } // qnode is leaf, rnode is leaf.
    else {
      typename ProblemType::TableType::TreeType *rnode_first;
      core::math::Range squared_distance_range_first,
           squared_distance_range_second;
      typename ProblemType::TableType::TreeType *rnode_second;
      Heuristic(
        metric, qnode, rnode->left(), rnode->right(),
        &rnode_first, squared_distance_range_first,
        &rnode_second, squared_distance_range_second);

      // Recurse.
      bool rnode_first_exact =
        DualtreeCanonical_(
          metric, qnode, rnode_first, failure_probability / 2.0,
          squared_distance_range_first, query_results);

      bool rnode_second_exact =
        DualtreeCanonical_(
          metric, qnode, rnode_second,
          (rnode_first_exact) ? failure_probability : failure_probability / 2.0,
          squared_distance_range_second, query_results);
      exact_compute = rnode_first_exact && rnode_second_exact;
    } // qnode is leaf, rnode is not leaf.
    return exact_compute;
  } // end of query node being a leaf.

  // If we are here, we have to split the query.
  bool exact_compute_nonleaf_qnode = true;

  // Get the current query node statistic.
  typename ProblemType::StatisticType &qnode_stat = qnode->stat();

  // Left and right nodes of the query node and their statistic.
  typename ProblemType::TableType::TreeType *qnode_left = qnode->left();
  typename ProblemType::TableType::TreeType *qnode_right = qnode->right();
  typename ProblemType::StatisticType &qnode_left_stat = qnode_left->stat();
  typename ProblemType::StatisticType &qnode_right_stat = qnode_right->stat();

  // Push down postponed and clear.
  qnode_left_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
  qnode_right_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
  qnode_stat.postponed_.SetZero();

  if(rnode->is_leaf()) {
    typename ProblemType::TableType::TreeType *qnode_first;
    core::math::Range
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::TableType::TreeType *qnode_second;
    Heuristic(
      metric, rnode, qnode_left, qnode_right,
      &qnode_first, squared_distance_range_first,
      &qnode_second, squared_distance_range_second);

    // Recurse.
    bool first_qnode_exact =
      DualtreeCanonical_(
        metric, qnode_first, rnode, failure_probability,
        squared_distance_range_first, query_results);
    bool second_qnode_exact =
      DualtreeCanonical_(
        metric, qnode_second, rnode, failure_probability,
        squared_distance_range_second, query_results);
    exact_compute_nonleaf_qnode = first_qnode_exact && second_qnode_exact;
  } // qnode is not leaf, rnode is leaf.

  else {
    typename ProblemType::TableType::TreeType *rnode_first;
    core::math::Range
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::TableType::TreeType *rnode_second;
    Heuristic(
      metric, qnode_left, rnode->left(), rnode->right(),
      &rnode_first, squared_distance_range_first,
      &rnode_second, squared_distance_range_second);

    // Recurse.
    bool qnode_left_rnode_first_exact =
      DualtreeCanonical_(
        metric, qnode_left, rnode_first, failure_probability / 2.0,
        squared_distance_range_first, query_results);
    bool qnode_left_rnode_second_exact =
      DualtreeCanonical_(
        metric, qnode_left, rnode_second,
        (qnode_left_rnode_first_exact) ?
        failure_probability : failure_probability / 2.0,
        squared_distance_range_second, query_results);

    Heuristic(
      metric, qnode_right, rnode->left(), rnode->right(),
      &rnode_first, squared_distance_range_first,
      &rnode_second, squared_distance_range_second);

    // Recurse.
    bool qnode_right_rnode_first_exact =
      DualtreeCanonical_(
        metric, qnode_right, rnode_first, failure_probability / 2.0,
        squared_distance_range_first, query_results);
    bool qnode_right_rnode_second_exact =
      DualtreeCanonical_(
        metric, qnode_right, rnode_second,
        (qnode_right_rnode_first_exact) ?
        failure_probability : failure_probability / 2.0,
        squared_distance_range_second, query_results);

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
template<typename MetricType>
void DualtreeDfs<ProblemType>::PostProcess_(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::ResultType *query_results,
  bool do_query_results_postprocess) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();

  if(qnode->is_leaf()) {

    typename ProblemType::TableType::TreeIterator qnode_iterator =
      query_table_->get_node_iterator(qnode);

    // Reset the summary statistics.
    qnode_stat.summary_.StartReaccumulate();

    while(qnode_iterator.HasNext()) {
      arma::vec q_col;
      int q_index;
      double q_weight;
      qnode_iterator.Next(&q_col, &q_index, &q_weight);
      query_results->FinalApplyPostponed(
        problem_->global(), q_col, q_index, qnode_stat.postponed_);

      if(do_query_results_postprocess) {
        query_results->PostProcess(
          metric, q_col, q_index, q_weight, problem_->global(),
          problem_->is_monochromatic());
      }

      // Refine min and max summary statistics.
      qnode_stat.summary_.Accumulate(
        problem_->global(), *query_results, q_index);
    }
    qnode_stat.postponed_.FinalSetZero();
  }
  else {
    typename ProblemType::TableType::TreeType *qnode_left = qnode->left();
    typename ProblemType::TableType::TreeType *qnode_right = qnode->right();
    typename ProblemType::StatisticType &qnode_left_stat = qnode_left->stat();
    typename ProblemType::StatisticType &qnode_right_stat = qnode_right->stat();

    // For the final term, push down the postponed contribution.
    qnode_left_stat.postponed_.FinalApplyPostponed(
      problem_->global(), qnode_stat.postponed_);
    qnode_right_stat.postponed_.FinalApplyPostponed(
      problem_->global(), qnode_stat.postponed_);
    qnode_stat.postponed_.FinalSetZero();

    // Recursively postprocess the left and the right results.
    PostProcess_(
      metric, qnode_left,  query_results, do_query_results_postprocess);
    PostProcess_(
      metric, qnode_right, query_results, do_query_results_postprocess);

    // Refine the summary statistics.
    qnode_stat.summary_.StartReaccumulate();
    qnode_stat.summary_.Accumulate(
      problem_->global(), qnode_left_stat.summary_,
      qnode_left_stat.postponed_);
    qnode_stat.summary_.Accumulate(
      problem_->global(), qnode_right_stat.summary_,
      qnode_right_stat.postponed_);
  }
}
}
}

#endif
