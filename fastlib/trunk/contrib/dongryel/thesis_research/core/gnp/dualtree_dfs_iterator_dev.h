/** @file dualtree_dfs_iterator_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_ITERATOR_DEV_H
#define CORE_GNP_DUALTREE_DFS_ITERATOR_DEV_H

#include "dualtree_dfs.h"

namespace core {
namespace gnp {
template<typename ProblemType>
DualtreeDfs<ProblemType>::iterator::IteratorArgType::IteratorArgType() {

  // Initialize the members.
  qnode_ = NULL;
  rnode_ = NULL;

  // Compute the range squared distances between the two nodes.
  squared_distance_range_.InitEmptySet();
}

template<typename ProblemType>
DualtreeDfs<ProblemType>::iterator::
IteratorArgType::IteratorArgType(const IteratorArgType &arg_in) {

  // Initialize the members.
  qnode_ = arg_in.qnode();
  rnode_ = arg_in.rnode();

  // Compute the range squared distances between the two nodes.
  squared_distance_range_ = arg_in.squared_distance_range();
}

template<typename ProblemType>
typename ProblemType::TableType::TreeType *DualtreeDfs<ProblemType>::iterator
::IteratorArgType::qnode() {
  return qnode_;
}

template<typename ProblemType>
typename ProblemType::TableType::TreeType *DualtreeDfs<ProblemType>::iterator
::IteratorArgType::qnode() const {
  return qnode_;
}

template<typename ProblemType>
typename ProblemType::TableType::TreeType *DualtreeDfs<ProblemType>::iterator
::IteratorArgType::rnode() {
  return rnode_;
}

template<typename ProblemType>
typename ProblemType::TableType::TreeType *DualtreeDfs<ProblemType>::iterator
::IteratorArgType::rnode() const {
  return rnode_;
}

template<typename ProblemType>
const core::math::Range &DualtreeDfs<ProblemType>::iterator
::IteratorArgType::squared_distance_range() const {
  return squared_distance_range_;
}

template<typename ProblemType>
DualtreeDfs<ProblemType>::iterator::IteratorArgType::IteratorArgType(
  const core::metric_kernels::AbstractMetric &metric_in,
  typename DualtreeDfs<ProblemType>::TableType *query_table_in,
  typename DualtreeDfs<ProblemType>::TableType::TreeType *qnode_in,
  typename DualtreeDfs<ProblemType>::TableType *reference_table_in,
  typename DualtreeDfs<ProblemType>::TableType::TreeType *rnode_in) {

  // Initialize the members.
  qnode_ = qnode_in;
  rnode_ = rnode_in;
  squared_distance_range_ =
    (query_table_in->get_node_bound(qnode_in)).RangeDistanceSq(
      metric_in,
      reference_table_in->get_node_bound(rnode_in));
}

template<typename ProblemType>
DualtreeDfs<ProblemType>::iterator::IteratorArgType::IteratorArgType(
  const core::metric_kernels::AbstractMetric &metric_in,
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
DualtreeDfs<ProblemType>::iterator::iterator(
  const core::metric_kernels::AbstractMetric &metric_in,
  DualtreeDfs<ProblemType> &engine_in,
  typename ProblemType::ResultType *query_results_in): metric_(metric_in) {

  engine_ = &engine_in;
  query_table_ = engine_->query_table();
  reference_table_ = engine_->reference_table();
  query_results_ = query_results_in;

  // Initialize an empty trace for the computation and the query
  // root/reference root pair into the trace.
  trace_.Init();
  trace_.push_back(IteratorArgType(metric_in, query_table_,
                                   query_table_->get_tree(),
                                   reference_table_,
                                   reference_table_->get_tree()));
}

template<typename ProblemType>
void DualtreeDfs<ProblemType>::iterator::operator++() {

  // Push a blank argument to the trace for making the exit phase.
  trace_.push_front(IteratorArgType());

  // Pop the next item to visit in the list.
  IteratorArgType args = trace_.back();
  trace_.pop_back();

  while (trace_.empty() == false && args.rnode() != NULL) {

    // Get the arguments.
    TreeType *qnode = args.qnode();
    TreeType *rnode = args.rnode();
    const core::math::Range &squared_distance_range =
      args.squared_distance_range();

    // Compute the delta change.
    typename ProblemType::Delta_t delta;
    delta.DeterministicCompute(metric_, engine_->problem_->global(),
                               qnode, rnode, squared_distance_range);
    bool prunable = engine_->CanSummarize_(qnode, rnode, delta,
                                           query_results_);

    if (prunable) {
      engine_->Summarize_(qnode, delta, query_results_);
    }
    else {

      // If the query node is leaf node,
      if (query_table_->node_is_leaf(qnode)) {

        // If the reference node is leaf node,
        if (reference_table_->node_is_leaf(rnode)) {
          engine_->DualtreeBase_(metric_, qnode, rnode, query_results_);
        }
        else {
          TreeType *rnode_first;
          core::math::Range squared_distance_range_first,
          squared_distance_range_second;
          TreeType *rnode_second;
          engine_->Heuristic_(metric_, qnode, query_table_,
                              reference_table_->get_node_left_child(rnode),
                              reference_table_->get_node_right_child(rnode),
                              reference_table_,
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
        TreeType *qnode_left = query_table_->get_node_left_child(qnode);
        TreeType *qnode_right = query_table_->get_node_right_child(qnode);

        // If the reference node is leaf node,
        if (reference_table_->node_is_leaf(rnode)) {

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
          TreeType *rnode_left = reference_table_->get_node_left_child(rnode);
          TreeType *rnode_right =
            reference_table_->get_node_right_child(rnode);

          // Prioritize on the left child of the query node.
          TreeType *rnode_first = NULL, *rnode_second = NULL;
          core::math::Range squared_distance_range_first;
          core::math::Range squared_distance_range_second;
          engine_->Heuristic_(metric_, qnode_left, query_table_, rnode_left,
                              rnode_right, reference_table_,
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
          engine_->Heuristic_(metric_, qnode_right, query_table_,
                              rnode_left, rnode_right, reference_table_,
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
void DualtreeDfs<ProblemType>::iterator::Finalize() {
  return engine_->PostProcess_(
           metric_, query_table_->get_tree(), query_results_);
}

template<typename ProblemType>
typename ProblemType::ResultType &DualtreeDfs<ProblemType>::iterator
::operator*() {
  return *query_results_;
}

template<typename ProblemType>
const typename ProblemType::ResultType &DualtreeDfs<ProblemType>::iterator
::operator*() const {
  return *query_results_;
}

template<typename ProblemType>
typename DualtreeDfs<ProblemType>::iterator
DualtreeDfs<ProblemType>::get_iterator(
  const core::metric_kernels::AbstractMetric &metric_in,
  typename ProblemType::ResultType *query_results_in) {

  // Allocate space for storing the final results.
  query_results_in->Init(query_table_->n_entries());

  return typename DualtreeDfs<ProblemType>::iterator(
           metric_in, *this, query_results_in);
}
};
};

#endif
