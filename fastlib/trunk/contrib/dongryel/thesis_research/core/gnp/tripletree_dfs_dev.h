/** @file tripletree_dfs_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_TRIPLETREE_DFS_DEV_H
#define CORE_GNP_TRIPLETREE_DFS_DEV_H

#include <armadillo>
#include "tripletree_dfs.h"
#include "core/table/table.h"
#include "core/gnp/triple_distance_sq.h"
#include "core/math/math_lib.h"

template<typename ProblemType>
int core::gnp::TripletreeDfs<ProblemType>::num_deterministic_prunes() const {
  return num_deterministic_prunes_;
}

template<typename ProblemType>
int core::gnp::TripletreeDfs<ProblemType>::num_monte_carlo_prunes() const {
  return num_monte_carlo_prunes_;
}

template<typename ProblemType>
ProblemType *core::gnp::TripletreeDfs<ProblemType>::problem() {
  return problem_;
}

template<typename ProblemType>
typename ProblemType::TableType
*core::gnp::TripletreeDfs<ProblemType>::table() {
  return table_;
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::ResetStatistic() {
  ResetStatisticRecursion_(table_->get_tree());
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::Init(ProblemType &problem_in) {
  problem_ = &problem_in;
  table_ = problem_->table();
  ResetStatistic();

  // Reset prune statistics.
  num_deterministic_prunes_ = 0;
  num_monte_carlo_prunes_ = 0;
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::NaiveCompute(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::ResultType *naive_query_results) {

  // Preprocess the tree.
  PreProcess_(table_->get_tree());

  // Allocate space for storing the final results.
  naive_query_results->Init(table_->n_entries());

  // Call the algorithm computation.
  std::vector< TreeType *> root_nodes(3, table_->get_tree());
  core::gnp::TripleRangeDistanceSq<TableType> triple_range_distance_sq;
  triple_range_distance_sq.Init(metric, *table_, root_nodes);
  TripletreeBase_(
    metric, triple_range_distance_sq, naive_query_results);
  PostProcess_(metric, table_->get_tree(), naive_query_results, true);
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::Compute(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::ResultType *query_results) {

  // Preprocess the tree.
  PreProcess_(table_->get_tree());

  // Allocate space for storing the final results.
  query_results->Init(table_->n_entries());

  // Call the algorithm computation.
  std::vector< TreeType *> root_nodes(3, table_->get_tree());
  core::gnp::TripleRangeDistanceSq<TableType> triple_range_distance_sq;
  triple_range_distance_sq.Init(metric, *table_, root_nodes);

  // Monochromatic computation.
  std::vector< typename TableType::TreeType *> leaf_nodes;
  table_->get_leaf_nodes(table_->get_tree(), &leaf_nodes);
  for(unsigned int i = 0; i < leaf_nodes.size(); i++) {
    std::vector< TreeType *> leaf_node_tuples(3, leaf_nodes[i]);
    core::gnp::TripleRangeDistanceSq<TableType> range_sq_in;
    range_sq_in.Init(metric, *table_, leaf_node_tuples);
    TripletreeBase_(
      metric, range_sq_in, query_results);
  }
  PostProcess_(metric, table_->get_tree(), query_results, false);

  std::vector<double> top_failure_probabilities(
    3, 1.0 - problem_->global().probability());
  TripletreeCanonical_(
    metric,
    triple_range_distance_sq,
    problem_->global().relative_error(),
    top_failure_probabilities,
    query_results);
  PostProcess_(metric, table_->get_tree(), query_results, true);
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::ResetStatisticRecursion_(
  typename ProblemType::TableType::TreeType *node) {
  table_->get_node_stat(node).SetZero();
  if(table_->node_is_leaf(node) == false) {
    ResetStatisticRecursion_(table_->get_node_left_child(node));
    ResetStatisticRecursion_(table_->get_node_right_child(node));
  }
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::PreProcess_(
  typename ProblemType::TableType::TreeType *qnode) {

  typename ProblemType::StatisticType &qnode_stat =
    table_->get_node_stat(qnode);
  typename ProblemType::TableType::TreeIterator qnode_it =
    table_->get_node_iterator(qnode);

  if(! table_->node_is_leaf(qnode)) {
    PreProcess_(table_->get_node_left_child(qnode));
    PreProcess_(table_->get_node_right_child(qnode));
    qnode_stat.Init(
      qnode_it, table_->get_node_left_child(qnode)->stat(),
      table_->get_node_right_child(qnode)->stat());
  }
  else {
    qnode_stat.Init(qnode_it);
  }
}

template<typename ProblemType>
typename core::gnp::TripletreeDfs<ProblemType>::TableType::TreeIterator
core::gnp::TripletreeDfs<ProblemType>::GetNextNodeIterator_(
  const core::gnp::TripleRangeDistanceSq<TableType> &range_sq_in,
  int node_index,
  const typename TableType::TreeIterator &it_in) {

  if(range_sq_in.node(node_index) != range_sq_in.node(node_index + 1)) {
    return table_->get_node_iterator(range_sq_in.node(node_index + 1));
  }
  else {
    return it_in;
  }
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::TripletreeBase_(
  const core::metric_kernels::AbstractMetric &metric,
  const core::gnp::TripleRangeDistanceSq<TableType> &range_sq_in,
  typename ProblemType::ResultType *query_results) {

  // Temporary postponed objects to be used within the triple loop.
  std::vector< typename ProblemType::PostponedType > point_postponeds;
  point_postponeds.resize(3);

  // The triple object used for keeping track of the squared
  // distances.
  core::gnp::TripleDistanceSq distance_sq_set;

  // Loop through the first node.
  typename TableType::TreeIterator first_node_it =
    table_->get_node_iterator(range_sq_in.node(0));
  do {
    // Get the point in the first node.
    core::table::DensePoint first_point;
    int first_point_index;
    first_node_it.Next(&first_point, &first_point_index);
    distance_sq_set.ReplaceOnePoint(metric, first_point, first_point_index, 0);

    // Construct the second iterator and start looping.
    typename TableType::TreeIterator second_node_it =
      GetNextNodeIterator_(range_sq_in, 0, first_node_it);
    while(second_node_it.HasNext()) {

      // Get the point in the second node.
      core::table::DensePoint second_point;
      int second_point_index;
      second_node_it.Next(&second_point, &second_point_index);
      distance_sq_set.ReplaceOnePoint(
        metric, second_point, second_point_index, 1);

      // Loop through the third node.
      typename TableType::TreeIterator third_node_it =
        GetNextNodeIterator_(range_sq_in, 1, second_node_it);
      while(third_node_it.HasNext()) {

        // Get thet point in the third node.
        core::table::DensePoint third_point;
        int third_point_index;
        third_node_it.Next(&third_point, &third_point_index);
        distance_sq_set.ReplaceOnePoint(
          metric, third_point, third_point_index, 2);

        // Add the contribution due to the triple that has been chosen
        // to each of the query point.
        problem_->global().ApplyContribution(
          distance_sq_set, &point_postponeds);

        // Apply the postponed contribution to each query result.
        query_results->ApplyPostponed(
          first_point_index, point_postponeds[0]);
        query_results->ApplyPostponed(
          second_point_index, point_postponeds[1]);
        query_results->ApplyPostponed(
          third_point_index, point_postponeds[2]);

      } // end of looping over the third node.
    } // end of looping over the second node.
  } // end of looping over the first node.
  while(first_node_it.HasNext());

  for(int node_index = 0; node_index < 3; node_index++) {
    if(node_index == 0 ||
        range_sq_in.node(node_index) != range_sq_in.node(node_index - 1)) {

      // Clear the summary statistics of the current query node so that we
      // can refine it to better bounds.
      typename ProblemType::TableType::TreeType *node =
        range_sq_in.node(node_index);
      typename ProblemType::StatisticType &node_stat =
        problem_->table()->get_node_stat(node);
      node_stat.summary_.StartReaccumulate(problem_->global());

      // Get the query node iterator and the reference node iterator.
      typename ProblemType::TableType::TreeIterator node_iterator =
        problem_->table()->get_node_iterator(node);

      // Add the pruned tuples at this base case to the postponed of
      // the current node (which will be all cleared when the function
      // is exited).
      node_stat.postponed_.pruned_ += range_sq_in.num_tuples(node_index);

      // Apply the postponed contribution to the each node.
      while(node_iterator.HasNext()) {

        // Get the query point and its real index.
        core::table::DensePoint q_col;
        int q_index;
        node_iterator.Next(&q_col, &q_index);

        // Incorporate the postponed information.
        query_results->ApplyPostponed(q_index, node_stat.postponed_);

        // Refine min and max summary statistics.
        node_stat.summary_.Accumulate(
          problem_->global(), *query_results, q_index);

      } // end of looping over each query point.

      // Postaccumulate operation.
      node_stat.summary_.PostAccumulate(problem_->global());

      // Clear postponed information.
      node_stat.postponed_.SetZero();
    }
  } // end of looping over each node.
}

template<typename ProblemType>
bool core::gnp::TripletreeDfs<ProblemType>::CanProbabilisticSummarize_(
  const core::metric_kernels::AbstractMetric &metric,
  const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
  const std::vector<double> &failure_probabilities,
  int node_start_index,
  typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  // Since we have turned on the monochromatic trick, fail if all
  // three nodes are equal.
  if(range_in.node(0) == range_in.node(1) &&
      range_in.node(1) == range_in.node(2)) {
    return false;
  }

  // If there are too many tuples to approximate, then don't try.
  if(std::max(
        range_in.num_tuples(0), std::max(
          range_in.num_tuples(1), range_in.num_tuples(2))) >=
      0.01 * problem_->global().total_num_tuples()) {
    return false;
  }

  // Prepare for Monte Carlo accumulation.
  delta.ResetMeanVariancePairs(
    problem_->global(), range_in.nodes(), node_start_index);

  // The summary statistics.
  typename ProblemType::SummaryType new_summary;

  bool flag = true;
  core::table::DensePoint previous_query_point;
  int previous_query_point_index = -1;

  for(int i = node_start_index; flag && i < 3; i++) {
    typename core::gnp::TripletreeDfs<ProblemType>::TreeType *node =
      range_in.node(i);
    if(i == 0 || node != range_in.node(i - 1)) {
      typename ProblemType::StatisticType &node_stat =
        table_->get_node_stat(node);

      // Loop over each point on this node.
      typename TableType::TreeIterator node_it =
        table_->get_node_iterator(node);

      // The query point index and the point.
      int query_point_index = -1;
      core::table::DensePoint query_point;

      // The new summary.
      new_summary = node_stat.summary_;
      new_summary.ApplyPostponed(node_stat.postponed_);

      for(int qpoint_dfs_index = node->begin(); node_it.HasNext() && flag;
          qpoint_dfs_index++) {

        // The current query point.
        node_it.Next(&query_point, &query_point_index);

        if(previous_query_point_index >= 0) {
          flag = new_summary.CanProbabilisticSummarize(
                   metric, problem_->global(), delta,
                   range_in, failure_probabilities, i, query_results,
                   query_point, qpoint_dfs_index, query_point_index,
                   &previous_query_point, &previous_query_point_index);
        }
        else {
          flag = new_summary.CanProbabilisticSummarize(
                   metric, problem_->global(), delta,
                   range_in, failure_probabilities, i, query_results,
                   query_point, qpoint_dfs_index, query_point_index,
                   (const core::table::DensePoint *) NULL,
                   (int *) NULL);
        }
        previous_query_point.Alias(query_point);
        previous_query_point_index = query_point_index;
      }
    }
  }

  return flag;
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::ProbabilisticSummarize_(
  const core::metric_kernels::AbstractMetric &metric,
  GlobalType &global,
  const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
  const std::vector<double> &failure_probabilities,
  int probabilistic_node_start_index,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  // Apply the deterministic prunes.
  Summarize_(
    range_in, probabilistic_node_start_index, delta, query_results);

  // Apply the probabilistic prunes.
  query_results->ApplyProbabilisticDelta(
    global, range_in, failure_probabilities,
    probabilistic_node_start_index, delta);

  // Do a full refine by traversing each node.
  for(int i = 0; i < 3; i++) {
    typename core::gnp::TripletreeDfs<ProblemType>::TreeType *node =
      range_in.node(i);
    if(i == 0 || range_in.node(i - 1) != node) {
      PostProcess_(metric, node, query_results, false);
    }
  }
}

template<typename ProblemType>
bool core::gnp::TripletreeDfs<ProblemType>::CanSummarize_(
  const core::gnp::TripleRangeDistanceSq<TableType>
  &triple_range_distance_sq_in,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results,
  int *failure_index) {

  // Since we have turned on the monochromatic trick, fail if all
  // three nodes are equal.
  if(triple_range_distance_sq_in.node(0) ==
      triple_range_distance_sq_in.node(1) &&
      triple_range_distance_sq_in.node(1) ==
      triple_range_distance_sq_in.node(2)) {
    return false;
  }

  std::vector< typename ProblemType::SummaryType > new_summaries;
  new_summaries.resize(3);

  bool flag = true;
  for(int i = 0; flag && i < 3; i++) {
    typename core::gnp::TripletreeDfs<ProblemType>::TreeType *node =
      triple_range_distance_sq_in.node(i);
    if(i == 0 || node != triple_range_distance_sq_in.node(i - 1)) {
      typename ProblemType::StatisticType &node_stat =
        table_->get_node_stat(node);
      new_summaries[i] = node_stat.summary_;
      new_summaries[i].ApplyPostponed(node_stat.postponed_);
      new_summaries[i].ApplyDelta(delta, i);
      flag = new_summaries[i].CanSummarize(
               problem_->global(), delta, triple_range_distance_sq_in, i,
               query_results);

      if(flag == false) {
        *failure_index = i;
      }
    }
  }
  return flag;
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::AllocateProbabilities_(
  const std::vector<double> &failure_probabilities,
  const std::deque<bool> &node_is_split,
  const std::deque<bool> &recurse_to_left,
  const std::vector<int> &deterministic_computation_count,
  std::vector<double> *new_failure_probabilities) const {

  new_failure_probabilities->resize(3);
  for(unsigned int i = 0; i < node_is_split.size(); i++) {
    int count = 0;
    for(unsigned int j = 0; j < node_is_split.size(); j++) {
      if(i != j && node_is_split[j]) {
        count++;
      }
    }
    int minus_count = (recurse_to_left[i]) ?
                      deterministic_computation_count[ 2 * i ] :
                      deterministic_computation_count[ 2 * i + 1];
    (*new_failure_probabilities)[i] = failure_probabilities[i] /
                                      static_cast<double>(
                                        (1 << count) - minus_count);
  }
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::Summarize_(
  const core::gnp::TripleRangeDistanceSq<TableType> &triple_range_distance_sq,
  int probabilistic_node_start_index,
  const typename ProblemType::DeltaType &delta,
  typename ProblemType::ResultType *query_results) {

  for(int i = 0; i < probabilistic_node_start_index; i++) {
    typename core::gnp::TripletreeDfs<ProblemType>::TreeType *node =
      triple_range_distance_sq.node(i);
    if(i == 0 || node != triple_range_distance_sq.node(i - 1)) {
      typename ProblemType::StatisticType &node_stat =
        table_->get_node_stat(node);
      node_stat.postponed_.ApplyDelta(delta, i, query_results);
    }
  }
}

template<typename ProblemType>
bool core::gnp::TripletreeDfs<ProblemType>::NodeIsAgreeable_(
  typename core::gnp::TripletreeDfs<ProblemType>::TreeType *node,
  typename core::gnp::TripletreeDfs<ProblemType>::TreeType *next_node) const {

  // Agreeable if the nodes are equal or the next node's beginning
  // index is more than the ending index of the given node.
  return node == next_node || node->end() <= next_node->begin();
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::RecursionHelper_(
  const core::metric_kernels::AbstractMetric &metric,
  core::gnp::TripleRangeDistanceSq<TableType> &triple_range_distance_sq,
  double relative_error,
  const std::vector<double> &failure_probabilities,
  typename ProblemType::ResultType *query_results,
  int level,
  bool all_leaves,
  std::deque<bool> &node_is_split,
  std::deque<bool> &recurse_to_left,
  std::vector<int> &deterministic_computation_count,
  bool *deterministic_approximation) {

  // If we have chosen all three nodes,
  if(level == 3) {

    bool exact_computation = true;
    if(all_leaves) {

      if(!(
            triple_range_distance_sq.node(0) ==
            triple_range_distance_sq.node(1) &&
            triple_range_distance_sq.node(1) ==
            triple_range_distance_sq.node(2))) {

        // Call the base case when all three nodes are leaves.
        TripletreeBase_(metric, triple_range_distance_sq, query_results);
      }
    }
    else {

      // Otherwise call the canonical case.
      std::vector<double> new_failure_probabilities;
      AllocateProbabilities_(
        failure_probabilities, node_is_split,
        recurse_to_left, deterministic_computation_count,
        &new_failure_probabilities);

      exact_computation =
        TripletreeCanonical_(
          metric, triple_range_distance_sq, relative_error,
          new_failure_probabilities, query_results);
      *deterministic_approximation = (*deterministic_approximation) &&
                                     exact_computation;
    }

    // Update the deterministic computation count so that the
    // probabilities can be redistributed.
    if(exact_computation) {
      for(int i = 0; i < 3; i++) {
        if(recurse_to_left[i]) {
          deterministic_computation_count[2 * i]++;
        }
        else {
          deterministic_computation_count[2 * i + 1]++;
        }
      }
    }
  }

  // Otherwise, keep choosing the nodes.
  else {

    TreeType *current_node = triple_range_distance_sq.node(level);

    // If the current node is a leaf node, then just check whether it
    // is in conflict with the previously chosen node.
    if(current_node->is_leaf()) {
      if(level == 0 ||
          NodeIsAgreeable_(
            triple_range_distance_sq.node(level - 1),
            current_node)) {

        recurse_to_left[level] = true;
        RecursionHelper_(
          metric, triple_range_distance_sq, relative_error,
          failure_probabilities, query_results,
          level + 1, all_leaves, node_is_split,
          recurse_to_left, deterministic_computation_count,
          deterministic_approximation);
      }
    }

    // Otherwise we need to split.
    else {

      // Node is split on the current level.
      node_is_split[level] = true;

      // Get the current query node statistic.
      typename ProblemType::StatisticType &current_node_stat =
        table_->get_node_stat(current_node);

      // Left and right nodes of the query node and their statistic.
      typename ProblemType::TableType::TreeType *current_node_left =
        table_->get_node_left_child(current_node);
      typename ProblemType::TableType::TreeType *current_node_right =
        table_->get_node_right_child(current_node);
      typename ProblemType::StatisticType &current_node_left_stat =
        table_->get_node_stat(current_node_left);
      typename ProblemType::StatisticType &current_node_right_stat =
        table_->get_node_stat(current_node_right);

      // Push down postponed and clear.
      current_node_left_stat.postponed_.ApplyPostponed(
        current_node_stat.postponed_);
      current_node_right_stat.postponed_.ApplyPostponed(
        current_node_stat.postponed_);
      current_node_stat.postponed_.SetZero();

      bool replaced_node_on_current_level = false;

      // Try the left child if it is valid.
      if(level == 0 ||
          NodeIsAgreeable_(
            triple_range_distance_sq.node(level - 1), current_node_left)) {

        replaced_node_on_current_level = true;
        triple_range_distance_sq.ReplaceOneNodeForward(
          metric, *table_, current_node_left, level);

        // Recursing to the left.
        recurse_to_left[level] = true;
        RecursionHelper_(
          metric, triple_range_distance_sq, relative_error,
          failure_probabilities, query_results, level + 1, false, node_is_split,
          recurse_to_left, deterministic_computation_count,
          deterministic_approximation);
      }

      // Try the right child if it is valid.
      if(level == 0 ||
          NodeIsAgreeable_(
            triple_range_distance_sq.node(level - 1), current_node_right)) {

        replaced_node_on_current_level = true;
        triple_range_distance_sq.ReplaceOneNodeForward(
          metric, *table_, current_node_right, level);

        // Recursing to the right.
        recurse_to_left[level] = false;
        RecursionHelper_(
          metric, triple_range_distance_sq, relative_error,
          failure_probabilities, query_results, level + 1, false, node_is_split,
          recurse_to_left, deterministic_computation_count,
          deterministic_approximation);
      }

      // Put back the node if it has been replaced before popping up
      // the recursion.
      if(replaced_node_on_current_level) {
        triple_range_distance_sq.ReplaceOneNodeBackward(
          metric, *table_, current_node, level);
      }

      // Need to refine the summary statistics by looking at the
      // children.
      current_node_stat.summary_.StartReaccumulate();
      current_node_stat.summary_.Accumulate(
        problem_->global(), current_node_left_stat.summary_,
        current_node_left_stat.postponed_);
      current_node_stat.summary_.Accumulate(
        problem_->global(), current_node_right_stat.summary_,
        current_node_right_stat.postponed_);

    } // end of the non-leaf case.
  } // end of choosing a node in each level.
}

template<typename ProblemType>
bool core::gnp::TripletreeDfs<ProblemType>::TripletreeCanonical_(
  const core::metric_kernels::AbstractMetric &metric,
  core::gnp::TripleRangeDistanceSq<TableType> &triple_range_distance_sq,
  double relative_error,
  const std::vector<double> &failure_probabilities,
  typename ProblemType::ResultType *query_results) {

  // Compute the delta.
  typename ProblemType::DeltaType delta;
  delta.DeterministicCompute(
    metric, problem_->global(), triple_range_distance_sq);

  int failure_index = 0;

  // First try to prune.
  if(CanSummarize_(
        triple_range_distance_sq, delta, query_results, &failure_index)) {
    Summarize_(
      triple_range_distance_sq, 3, delta, query_results);
    num_deterministic_prunes_++;
    return true;
  }

  // Then try probabilistic approximation.
  else if(
    failure_probabilities[0] > 0 &&
    failure_probabilities[1] > 0 &&
    failure_probabilities[2] > 0 &&
    CanProbabilisticSummarize_(
      metric, triple_range_distance_sq, failure_probabilities, failure_index,
      delta, query_results)) {
    ProbabilisticSummarize_(
      metric, problem_->global(), triple_range_distance_sq,
      failure_probabilities, failure_index, delta, query_results);
    num_monte_carlo_prunes_++;
    return false;
  }

  // Call the recursion helper.
  bool deterministic_approximation = true;
  std::deque<bool> node_is_split(3, false);
  std::deque<bool> recurse_to_left(3, true);
  std::vector<int> deterministic_computation_count(6, 0);
  RecursionHelper_(
    metric, triple_range_distance_sq, relative_error,
    failure_probabilities, query_results,
    0, true, node_is_split, recurse_to_left,
    deterministic_computation_count, &deterministic_approximation);

  return deterministic_approximation;
}

template<typename ProblemType>
void core::gnp::TripletreeDfs<ProblemType>::PostProcess_(
  const core::metric_kernels::AbstractMetric &metric,
  typename ProblemType::TableType::TreeType *qnode,
  typename ProblemType::ResultType *query_results,
  bool do_query_results_postprocess) {

  typename ProblemType::StatisticType &qnode_stat =
    table_->get_node_stat(qnode);

  if(table_->node_is_leaf(qnode)) {

    typename ProblemType::TableType::TreeIterator qnode_iterator =
      table_->get_node_iterator(qnode);

    // Reset the summary statistics.
    qnode_stat.summary_.StartReaccumulate(problem_->global());

    while(qnode_iterator.HasNext()) {
      core::table::DensePoint q_col;
      int q_index;
      qnode_iterator.Next(&q_col, &q_index);
      query_results->ApplyPostponed(q_index, qnode_stat.postponed_);

      if(do_query_results_postprocess) {
        query_results->PostProcess(metric, q_index, problem_->global());
      }

      // Refine min and max summary statistics.
      qnode_stat.summary_.Accumulate(
        problem_->global(), *query_results, q_index);
    }

    // Do post accumulate operation.
    qnode_stat.summary_.PostAccumulate(problem_->global());

    // Clear the postponed for the leaf node.
    qnode_stat.postponed_.SetZero();
  }
  else {
    typename ProblemType::TableType::TreeType *qnode_left =
      table_->get_node_left_child(qnode);
    typename ProblemType::TableType::TreeType *qnode_right =
      table_->get_node_right_child(qnode);
    typename ProblemType::StatisticType &qnode_left_stat =
      table_->get_node_stat(qnode_left);
    typename ProblemType::StatisticType &qnode_right_stat =
      table_->get_node_stat(qnode_right);

    qnode_left_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
    qnode_right_stat.postponed_.ApplyPostponed(qnode_stat.postponed_);
    qnode_stat.postponed_.SetZero();

    // Recurse to the left and to the right.
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

#endif
