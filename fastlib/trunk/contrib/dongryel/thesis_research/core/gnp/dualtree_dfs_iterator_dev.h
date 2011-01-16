/** @file dualtree_dfs_iterator_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_ITERATOR_DEV_H
#define CORE_GNP_DUALTREE_DFS_ITERATOR_DEV_H

#include "core/gnp/dualtree_dfs.h"

namespace core {
namespace gnp {
template<typename ProblemType>
template<typename IteratorMetricType>
DualtreeDfs<ProblemType>::iterator <
IteratorMetricType >::IteratorArgType::IteratorArgType() {

  // Initialize the members.
  qnode_ = NULL;
  rnode_ = NULL;

  // Compute the range squared distances between the two nodes.
  squared_distance_range_.InitEmptySet();
}

template<typename ProblemType>
template<typename IteratorMetricType>
DualtreeDfs<ProblemType>::iterator<IteratorMetricType>::
IteratorArgType::IteratorArgType(const IteratorArgType &arg_in) {

  // Initialize the members.
  qnode_ = arg_in.qnode();
  rnode_ = arg_in.rnode();

  // Compute the range squared distances between the two nodes.
  squared_distance_range_ = arg_in.squared_distance_range();
}

template<typename ProblemType>
template<typename IteratorMetricType>
typename ProblemType::TableType::TreeType *DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>::IteratorArgType::qnode() {
  return qnode_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
typename ProblemType::TableType::TreeType *DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>::IteratorArgType::qnode() const {
  return qnode_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
typename ProblemType::TableType::TreeType *DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>::IteratorArgType::rnode() {
  return rnode_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
typename ProblemType::TableType::TreeType *DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>::IteratorArgType::rnode() const {
  return rnode_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
const core::math::Range &DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>
::IteratorArgType::squared_distance_range() const {
  return squared_distance_range_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
DualtreeDfs<ProblemType>::iterator <
IteratorMetricType >::IteratorArgType::IteratorArgType(
  const IteratorMetricType &metric_in,
  typename DualtreeDfs<ProblemType>::TableType *query_table_in,
  typename DualtreeDfs<ProblemType>::TableType::TreeType *qnode_in,
  typename DualtreeDfs<ProblemType>::TableType *reference_table_in,
  typename DualtreeDfs<ProblemType>::TableType::TreeType *rnode_in) {

  // Initialize the members.
  qnode_ = qnode_in;
  rnode_ = rnode_in;
  squared_distance_range_ =
    (qnode_in->bound()).RangeDistanceSq(metric_in, rnode_in->bound());
}

template<typename ProblemType>
template<typename IteratorMetricType>
DualtreeDfs<ProblemType>::iterator <
IteratorMetricType >::IteratorArgType::IteratorArgType(
  const IteratorMetricType &metric_in,
  typename DualtreeDfs<ProblemType>::TableType *query_table_in,
  typename DualtreeDfs<ProblemType>::TableType::TreeType *qnode_in,
  typename DualtreeDfs<ProblemType>::TableType *reference_table_in,
  typename DualtreeDfs<ProblemType>::TableType::TreeType *rnode_in,
  const core::math::Range &squared_distance_range_in) {

  // Initialize the members.
  qnode_ = qnode_in;
  rnode_ = rnode_in;
  squared_distance_range_ = squared_distance_range_in;
}

template<typename ProblemType>
template<typename IteratorMetricType>
DualtreeDfs<ProblemType>::iterator<IteratorMetricType>::iterator(
  const IteratorMetricType &metric_in,
  DualtreeDfs<ProblemType> &engine_in,
  typename ProblemType::ResultType *query_results_in): metric_(metric_in) {

  engine_ = &engine_in;
  query_table_ = engine_->query_table();
  reference_table_ = engine_->reference_table();
  query_results_ = query_results_in;

  // Initialize an empty trace for the computation and the query
  // root/reference root pair into the trace.
  trace_.Init();
  trace_.push_back(IteratorArgType(
                     metric_in, query_table_,
                     query_table_->get_tree(),
                     reference_table_,
                     reference_table_->get_tree()));
}

template<typename ProblemType>
template<typename IteratorMetricType>
void DualtreeDfs<ProblemType>::iterator<IteratorMetricType>::operator++() {

  // Push a blank argument to the trace for making the exit phase.
  trace_.push_front(IteratorArgType());

  // Pop the next item to visit in the list.
  IteratorArgType args = trace_.back();
  trace_.pop_back();

  while(trace_.empty() == false && args.rnode() != NULL) {

    // Get the arguments.
    TreeType *qnode = args.qnode();
    TreeType *rnode = args.rnode();
    const core::math::Range &squared_distance_range =
      args.squared_distance_range();

    // Compute the delta change.
    typename ProblemType::DeltaType delta;
    delta.DeterministicCompute(
      metric_, engine_->problem_->global(),
      qnode, rnode, squared_distance_range);
    bool prunable = engine_->CanSummarize_(
                      qnode, rnode, delta, query_results_);

    if(prunable) {
      engine_->Summarize_(qnode, delta, query_results_);
    }
    else {

      // If the query node is leaf node,
      if(qnode->is_leaf()) {

        // If the reference node is leaf node,
        if(rnode->is_leaf()) {
          engine_->DualtreeBase_(metric_, qnode, rnode, query_results_);
        }
        else {
          TreeType *rnode_first;
          core::math::Range squared_distance_range_first,
               squared_distance_range_second;
          TreeType *rnode_second;
          engine_->Heuristic_(
            metric_, qnode, rnode->left(), rnode->right(),
            &rnode_first, squared_distance_range_first,
            &rnode_second, squared_distance_range_second);

          // Push the first prioritized reference node on the back
          // of the trace and the later one on the front of the
          // trace.
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode,
                             reference_table_, rnode_first,
                             squared_distance_range_first));
          trace_.push_front(IteratorArgType(
                              metric_, query_table_, qnode,
                              reference_table_, rnode_second,
                              squared_distance_range_second));
        }
      }

      // If the query node is a non-leaf node,
      else {

        // Here we split the query.
        TreeType *qnode_left = qnode->left();
        TreeType *qnode_right = qnode->right();

        // If the reference node is leaf node,
        if(rnode->is_leaf()) {

          // Push both combinations on the back of the trace.
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_left,
                             reference_table_, rnode));
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_right,
                             reference_table_, rnode));
        }

        // Otherwise, we split both the query and the reference.
        else {

          // Split the reference.
          TreeType *rnode_left = rnode->left();
          TreeType *rnode_right = rnode->right();

          // Prioritize on the left child of the query node.
          TreeType *rnode_first = NULL, *rnode_second = NULL;
          core::math::Range squared_distance_range_first;
          core::math::Range squared_distance_range_second;
          engine_->Heuristic_(
            metric_, qnode_left, rnode_left, rnode_right,
            &rnode_first, squared_distance_range_first,
            &rnode_second, squared_distance_range_second);
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_left,
                             reference_table_, rnode_first,
                             squared_distance_range_first));
          trace_.push_front(IteratorArgType(
                              metric_, query_table_, qnode_left,
                              reference_table_, rnode_second,
                              squared_distance_range_second));

          // Prioritize on the right child of the query node.
          engine_->Heuristic_(
            metric_, qnode_right, rnode_left, rnode_right,
            &rnode_first, squared_distance_range_first,
            &rnode_second, squared_distance_range_second);
          trace_.push_back(IteratorArgType(
                             metric_, query_table_, qnode_right,
                             reference_table_, rnode_first,
                             squared_distance_range_first));
          trace_.push_front(IteratorArgType(
                              metric_, query_table_, qnode_right,
                              reference_table_, rnode_second,
                              squared_distance_range_second));

        } // end of non-leaf query, non-leaf reference.
      } // end of non-leaf query.

    } // end of non-prunable case.

    // Pop the next item in the list.
    args = trace_.back();
    trace_.pop_back();

  } // end of the while loop.
}

template<typename ProblemType>
template<typename IteratorMetricType>
void DualtreeDfs<ProblemType>::iterator<IteratorMetricType>::Finalize() {
  return engine_->PostProcess_(
           metric_, query_table_->get_tree(), query_results_);
}

template<typename ProblemType>
template<typename IteratorMetricType>
typename ProblemType::ResultType &DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>::operator*() {
  return *query_results_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
const typename ProblemType::ResultType &DualtreeDfs <
ProblemType >::iterator<IteratorMetricType>::operator*() const {
  return *query_results_;
}

template<typename ProblemType>
template<typename IteratorMetricType>
typename DualtreeDfs<ProblemType>::template iterator<IteratorMetricType>
DualtreeDfs<ProblemType>::get_iterator(
  const IteratorMetricType &metric_in,
  typename ProblemType::ResultType *query_results_in) {

  // Allocate space for storing the final results.
  query_results_in->Init(query_table_->n_entries());

  return typename DualtreeDfs<ProblemType>::template
         iterator<IteratorMetricType>(
           metric_in, *this, query_results_in);
}
};
};

#endif
