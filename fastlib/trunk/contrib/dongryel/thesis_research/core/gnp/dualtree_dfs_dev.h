/** @file dualtree_dfs_dev.h
 *
 *  An implementation of the template generator for dualtree problems.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_DEV_H
#define CORE_GNP_DUALTREE_DFS_DEV_H

#include "dualtree_dfs.h"
#include "dualtree_dfs_iterator_dev.h"
#include "core/table/table.h"

namespace core {
namespace gnp {

template<typename ProblemType>
void DualtreeDfs<ProblemType>::set_query_reference_process_ranks(
  int query_process_id, int reference_process_id) {

  query_rank_ = query_process_id;
  reference_rank_ = reference_process_id;
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::set_query_start_node(
  TreeType *query_start_node_in, int qnode_recursion_depth_limit_in) {
  query_start_node_ = query_start_node_in;
  qnode_recursion_depth_limit_ = qnode_recursion_depth_limit_in;
}

template<typename ProblemType>
const std::vector <
typename core::gnp::DualtreeDfs<ProblemType>::FrontierObjectType >
&DualtreeDfs <
ProblemType >::unpruned_query_reference_pairs() const {
  return unpruned_query_reference_pairs_;
}

template<typename ProblemType>
template<typename PointSerializeFlagType>
void DualtreeDfs<ProblemType>::set_base_case_flags(
  const std::vector<PointSerializeFlagType> &flags_in) {

  for(unsigned int i = 0; i < flags_in.size(); i++) {
    serialize_points_per_terminal_node_[flags_in[i].begin()] =
      flags_in[i].count();
  }
  do_selective_base_case_ = true;
}

template<typename ProblemType>
DualtreeDfs<ProblemType>::DualtreeDfs() {
  qnode_recursion_depth_limit_ = std::numeric_limits<int>::max();
  do_selective_base_case_ = false;
  query_start_node_ = NULL;
  query_rank_ = 0;
  reference_rank_ = 0;
}

template<typename ProblemType>
int DualtreeDfs<ProblemType>::num_deterministic_prunes() const {
  return num_deterministic_prunes_;
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

  // Set the problem.
  problem_ = &problem_in;

  // Set the query table and its starting node.
  query_table_ = problem_->query_table();
  query_start_node_ = query_table_->get_tree();

  // Set the reference table.
  reference_table_ = problem_->reference_table();
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::Compute(
  const MetricType &metric,
  typename ProblemType::ResultType *query_results,
  bool do_initializations) {

  // Allocate space for storing the final results.
  if(do_initializations) {
    query_results->Init(query_table_->n_entries());
  }

  // Call the algorithm computation.
  core::math::Range squared_distance_range =
    (query_start_node_->bound()).RangeDistanceSq(
      metric, reference_table_->get_tree()->bound());

  if(do_initializations) {
    PreProcess_(query_start_node_);
  }
  PreProcessReferenceTree_(reference_table_->get_tree());

  DualtreeCanonical_(
    metric, query_start_node_, 0, reference_table_->get_tree(),
    1.0 - problem_->global().probability(), squared_distance_range,
    query_results);
  PostProcess_(metric, query_start_node_, query_results);

  printf("For reference process %d, query process %d pruned %d times.\n",
         reference_rank_, query_rank_, num_deterministic_prunes_);
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::PreProcessReferenceTree_(
  typename ProblemType::TableType::TreeType *rnode) {

  typename ProblemType::StatisticType &rnode_stat = rnode->stat();
  typename ProblemType::TableType::TreeIterator rnode_it =
    reference_table_->get_node_iterator(rnode);

  if(rnode->is_leaf()) {
    rnode_stat.Init(rnode_it);
  }
  else {

    // Get the left and the right children.
    typename ProblemType::TableType::TreeType *rnode_left_child = rnode->left();
    typename ProblemType::TableType::TreeType *rnode_right_child =
      rnode->right();

    // Recurse to the left and the right.
    PreProcessReferenceTree_(rnode_left_child);
    PreProcessReferenceTree_(rnode_right_child);

    // Build the node stat by combining those owned by the children.
    typename ProblemType::StatisticType &rnode_left_child_stat =
      rnode_left_child->stat();
    typename ProblemType::StatisticType &rnode_right_child_stat =
      rnode_right_child->stat();
    rnode_stat.Init(
      rnode_it, rnode_left_child_stat, rnode_right_child_stat);
  }
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::PreProcess_(
  typename ProblemType::TableType::TreeType *qnode) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.SetZero();

  if(! qnode->is_leaf()) {
    PreProcess_(qnode->left());
    PreProcess_(qnode->right());
  }
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::DualtreeBase_(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  typename ProblemType::ResultType *query_results) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
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
    core::table::DensePoint q_col;
    int q_index;
    qnode_iterator.Next(&q_col, &q_index);

    // Reset the temporary variable for accumulating each
    // reference point contribution.
    query_contribution.Init(problem_->global(), qnode, rnode);

    // Incorporate the postponed information.
    query_results->ApplyPostponed(q_index, qnode_stat.postponed_);

    // Reset the reference node iterator.
    rnode_iterator.Reset();
    while(rnode_iterator.HasNext()) {

      // Get the reference point and accumulate contribution.
      core::table::DensePoint r_col;
      int r_col_id;
      rnode_iterator.Next(&r_col, &r_col_id);
      if(
        problem_->global().is_monochromatic() == false ||
        r_col_id != q_index) {
        query_contribution.ApplyContribution(
          problem_->global(), metric, q_col, r_col);
      }

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
template<typename MetricType>
bool DualtreeDfs<ProblemType>::CanProbabilisticSummarize_(
  const MetricType &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  double failure_probability,
  typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  typename ProblemType::SummaryType new_summary(qnode_stat.summary_);
  new_summary.ApplyPostponed(qnode_stat.postponed_);
  new_summary.ApplyDelta(delta);

  return new_summary.CanProbabilisticSummarize(
           metric, problem_->global(), delta, qnode, rnode,
           failure_probability, query_results);
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::ProbabilisticSummarize_(
  typename ProblemType::GlobalType &global,
  typename ProblemType::TableType::TreeType *qnode,
  double failure_probability,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  query_results->ApplyProbabilisticDelta(
    global, qnode, failure_probability, delta);
}

template<typename ProblemType>
bool DualtreeDfs<ProblemType>::CanSummarize_(
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::TableType::TreeType *rnode,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  typename ProblemType::SummaryType new_summary(qnode_stat.summary_);
  new_summary.ApplyPostponed(qnode_stat.postponed_);
  new_summary.ApplyDelta(delta);

  return new_summary.CanSummarize(
           problem_->global(), delta, qnode, rnode, query_results);
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::Summarize_(
  typename ProblemType::TableType::TreeType *qnode,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();
  qnode_stat.postponed_.ApplyDelta(delta, query_results);
}

template<typename ProblemType>
template<typename MetricType>
void DualtreeDfs<ProblemType>::Heuristic_(
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
  int qnode_recursion_depth,
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
    num_deterministic_prunes_++;
    return true;
  }
  else if(failure_probability > 1e-6) {

    // Try Monte Carlo.
    if(CanProbabilisticSummarize_(
          metric, qnode, rnode, failure_probability, delta, query_results)) {
      ProbabilisticSummarize_(
        problem_->global(), qnode, failure_probability, delta, query_results);
      return false;
    }
  }

  // If it is not prunable and the query node is a leaf,
  if(qnode->is_leaf() ||
      qnode_recursion_depth >= qnode_recursion_depth_limit_) {

    bool exact_compute = true;
    if(rnode->is_leaf()) {

      if(do_selective_base_case_ == false ||
          serialize_points_per_terminal_node_.find(rnode->begin())
          != serialize_points_per_terminal_node_.end()) {

        // If the base case must be done, then do so.
        DualtreeBase_(metric, qnode, rnode, query_results);
      }
      else {

        // Otherwise, push into the list of reference nodes that must
        // be dealt later. These list will be used in the distributed
        // dualtree computation.
        double priority = squared_distance_range.lo /
                          static_cast<double>(qnode->count() * rnode->count());
        unpruned_query_reference_pairs_.push_back(
          boost::make_tuple(
            qnode,
            boost::make_tuple<int, int, int>(
              reference_rank_, rnode->begin(), rnode->count()), priority));
      }

    } // qnode is leaf, rnode is leaf.
    else {
      typename ProblemType::TableType::TreeType *rnode_first;
      core::math::Range squared_distance_range_first,
           squared_distance_range_second;
      typename ProblemType::TableType::TreeType *rnode_second;
      Heuristic_(
        metric, qnode, rnode->left(), rnode->right(),
        &rnode_first, squared_distance_range_first,
        &rnode_second, squared_distance_range_second);

      // Recurse.
      bool rnode_first_exact =
        DualtreeCanonical_(
          metric, qnode, qnode_recursion_depth,
          rnode_first, failure_probability / 2.0,
          squared_distance_range_first, query_results);

      bool rnode_second_exact =
        DualtreeCanonical_(
          metric, qnode, qnode_recursion_depth, rnode_second,
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
    Heuristic_(
      metric, rnode, qnode_left, qnode_right,
      &qnode_first, squared_distance_range_first,
      &qnode_second, squared_distance_range_second);

    // Recurse.
    bool first_qnode_exact =
      DualtreeCanonical_(
        metric, qnode_first, qnode_recursion_depth + 1,
        rnode, failure_probability,
        squared_distance_range_first, query_results);
    bool second_qnode_exact =
      DualtreeCanonical_(
        metric, qnode_second, qnode_recursion_depth + 1,
        rnode, failure_probability,
        squared_distance_range_second, query_results);
    exact_compute_nonleaf_qnode = first_qnode_exact && second_qnode_exact;
  } // qnode is not leaf, rnode is leaf.

  else {
    typename ProblemType::TableType::TreeType *rnode_first;
    core::math::Range
    squared_distance_range_first, squared_distance_range_second;
    typename ProblemType::TableType::TreeType *rnode_second;
    Heuristic_(
      metric, qnode_left, rnode->left(), rnode->right(),
      &rnode_first, squared_distance_range_first,
      &rnode_second, squared_distance_range_second);

    // Recurse.
    bool qnode_left_rnode_first_exact =
      DualtreeCanonical_(
        metric, qnode_left, qnode_recursion_depth + 1,
        rnode_first, failure_probability / 2.0,
        squared_distance_range_first, query_results);
    bool qnode_left_rnode_second_exact =
      DualtreeCanonical_(
        metric, qnode_left, qnode_recursion_depth + 1, rnode_second,
        (qnode_left_rnode_first_exact) ?
        failure_probability : failure_probability / 2.0,
        squared_distance_range_second, query_results);

    Heuristic_(
      metric, qnode_right, rnode->left(), rnode->right(),
      &rnode_first, squared_distance_range_first,
      &rnode_second, squared_distance_range_second);

    // Recurse.
    bool qnode_right_rnode_first_exact =
      DualtreeCanonical_(
        metric, qnode_right, qnode_recursion_depth + 1,
        rnode_first, failure_probability / 2.0,
        squared_distance_range_first, query_results);
    bool qnode_right_rnode_second_exact =
      DualtreeCanonical_(
        metric, qnode_right, qnode_recursion_depth + 1, rnode_second,
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
  typename ProblemType::ResultType *query_results) {

  typename ProblemType::StatisticType &qnode_stat = qnode->stat();

  if(qnode->is_leaf()) {

    typename ProblemType::TableType::TreeIterator qnode_iterator =
      query_table_->get_node_iterator(qnode);

    // Reset the summary statistics.
    qnode_stat.summary_.StartReaccumulate();

    while(qnode_iterator.HasNext()) {
      core::table::DensePoint q_col;
      int q_index;
      qnode_iterator.Next(&q_col, &q_index);
      query_results->ApplyPostponed(q_index, qnode_stat.postponed_);
      query_results->PostProcess(metric, q_index, problem_->global(),
                                 problem_->is_monochromatic());

      // Refine min and max summary statistics.
      qnode_stat.summary_.Accumulate(
        problem_->global(), *query_results, q_index);
    }
    qnode_stat.postponed_.SetZero();
  }
  else {
    typename ProblemType::TableType::TreeType *qnode_left = qnode->left();
    typename ProblemType::TableType::TreeType *qnode_right = qnode->right();
    typename ProblemType::StatisticType &qnode_left_stat = qnode_left->stat();
    typename ProblemType::StatisticType &qnode_right_stat = qnode_right->stat();

    qnode_left_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
    qnode_right_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
    qnode_stat.postponed_.SetZero();

    PostProcess_(metric, qnode_left,  query_results);
    PostProcess_(metric, qnode_right, query_results);

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
