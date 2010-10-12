/** @file gen_metric_tree_impl.h
 *
 *  Implementation for the regular pointer-style ball-tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <armadillo>
#include <deque>
#include "core/metric_kernels/abstract_metric.h"
#include "core/table/dense_matrix.h"

namespace core {
namespace tree_private {

// This function assumes that we have points embedded in Euclidean
// space.
template<typename TBound>
void MakeLeafMetricTreeNode(
  const core::metric_kernels::AbstractMetric &metric_in,
  const core::table::DenseMatrix& matrix,
  int begin, int count, TBound *bounds) {

  bounds->center().SetZero();

  int end = begin + count;
  for(int i = begin; i < end; i++) {
    bounds->center() += matrix.col(i);
  }
  bounds->center() /= ((double) count);

  double furthest_distance;
  FurthestColumnIndex(
    metric_in, bounds->center(), matrix, begin, count, &furthest_distance);
  bounds->set_radius(furthest_distance);
}

template<typename TBound>
int MatrixPartition(
  const core::metric_kernels::AbstractMetric &metric_in,
  core::table::DenseMatrix& matrix, int first, int count,
  TBound &left_bound, TBound &right_bound, std::vector<int> *old_from_new) {

  int end = first + count;
  int left_count = 0;

  std::deque<bool> left_membership;
  left_membership.resize(count);

  for(int left = first; left < end; left++) {

    // Make alias of the current point.
    core::table::DensePoint point;
    matrix.MakeColumnVector(left, &point);

    // Compute the distances from the two pivots.
    double distance_from_left_pivot =
      metric_in.Distance(point, left_bound.center());
    double distance_from_right_pivot =
      metric_in.Distance(point, right_bound.center());

    // We swap if the point is further away from the left pivot.
    if(distance_from_left_pivot > distance_from_right_pivot) {
      left_membership[left - first] = false;
    }
    else {
      left_membership[left - first] = true;
      left_count++;
    }
  }

  int left = first;
  int right = first + count - 1;

  // At any point: everything < left is correct
  //               everything > right is correct
  for(;;) {
    while(left_membership[left - first] && left <= right) {
      left++;
    }

    while(!left_membership[right - first] && left <= right) {
      right--;
    }

    if(left > right) {

      // left == right + 1
      break;
    }

    // Swap the left vector with the right vector.
    matrix.swap_cols(left, right);
    std::swap(left_membership[left - first], left_membership[right - first]);

    if(old_from_new) {
      std::swap((*old_from_new)[left], (*old_from_new)[right]);
    }
    right--;
  }

  return left_count;
}

int FurthestColumnIndex(
  const core::metric_kernels::AbstractMetric &metric_in,
  const core::table::AbstractPoint &pivot,
  const core::table::DenseMatrix &matrix,
  int begin, int count,
  double *furthest_distance) {

  int furthest_index = -1;
  int end = begin + count;
  *furthest_distance = -1.0;

  for(int i = begin; i < end; i++) {
    core::table::DensePoint point;
    matrix.MakeColumnVector(i, &point);
    double distance_between_center_and_point = metric_in.Distance(pivot, point);

    if((*furthest_distance) < distance_between_center_and_point) {
      *furthest_distance = distance_between_center_and_point;
      furthest_index = i;
    }
  }

  return furthest_index;
}

template<typename TMetricTree>
bool AttemptSplitting(
  const core::metric_kernels::AbstractMetric &metric_in,
  core::table::DenseMatrix& matrix, TMetricTree *node, TMetricTree **left,
  TMetricTree **right, int leaf_size,
  std::vector<int> *old_from_new) {

  // Pick a random row.
  int random_row = core::math::RandInt(
                     node->begin(), node->begin() + node->count());
  core::table::DensePoint random_row_vec;
  matrix.MakeColumnVector(random_row, & random_row_vec);

  // Now figure out the furthest point from the random row picked
  // above.
  double furthest_distance;
  int furthest_from_random_row =
    FurthestColumnIndex(
      metric_in, random_row_vec, matrix, node->begin(), node->count(),
      &furthest_distance);
  core::table::DensePoint furthest_from_random_row_vec;
  matrix.MakeColumnVector(
    furthest_from_random_row, &furthest_from_random_row_vec);

  // Then figure out the furthest point from the furthest point.
  double furthest_from_furthest_distance;
  int furthest_from_furthest_random_row =
    FurthestColumnIndex(
      metric_in, furthest_from_random_row_vec, matrix, node->begin(),
      node->count(), &furthest_from_furthest_distance);
  core::table::DensePoint furthest_from_furthest_random_row_vec;
  matrix.MakeColumnVector(
    furthest_from_furthest_random_row,
    &furthest_from_furthest_random_row_vec);

  if(furthest_from_furthest_distance <
      std::numeric_limits<double>::epsilon()) {
    return false;
  }
  else {
    *left = new TMetricTree();
    *right = new TMetricTree();

    ((*left)->bound().center()).Copy(furthest_from_random_row_vec);
    ((*right)->bound().center()).Copy(furthest_from_furthest_random_row_vec);

    int left_count = MatrixPartition(
                       metric_in, matrix, node->begin(), node->count(),
                       (*left)->bound(), (*right)->bound(), old_from_new);

    (*left)->Init(node->begin(), left_count);
    (*right)->Init(node->begin() + left_count, node->count() - left_count);
  }

  return true;
}

template<typename TMetricTree>
void CombineBounds(
  const core::metric_kernels::AbstractMetric &metric_in,
  core::table::DenseMatrix &matrix,
  TMetricTree *node, TMetricTree *left, TMetricTree *right) {

  // Compute the weighted sum of the two pivots
  node->bound().center().reference() =
    left->count() * left->bound().center().reference();
  node->bound().center().reference() = node->bound().center().reference() +
                                       right->count() *
                                       right->bound().center().reference();
  node->bound().center().reference() /= ((double) node->count());

  double left_max_dist, right_max_dist;
  FurthestColumnIndex(
    metric_in, node->bound().center(), matrix, left->begin(),
    left->count(), &left_max_dist);
  FurthestColumnIndex(
    metric_in, node->bound().center(), matrix, right->begin(),
    right->count(), &right_max_dist);
  node->bound().set_radius(std::max(left_max_dist, right_max_dist));
}

template<typename TMetricTree>
void SplitGenMetricTree(
  const core::metric_kernels::AbstractMetric &metric_in,
  core::table::DenseMatrix& matrix, TMetricTree *node,
  int leaf_size,
  int max_num_leaf_nodes,
  int *current_num_leaf_nodes,
  std::vector<int> *old_from_new,
  int *num_nodes) {

  TMetricTree *left = NULL;
  TMetricTree *right = NULL;

  // If the node is just too small or we have reached the maximum
  // number of leaf nodes allowed, then do not split.
  if(node->count() < leaf_size ||
      (*current_num_leaf_nodes) >= max_num_leaf_nodes) {
    MakeLeafMetricTreeNode(
      metric_in, matrix, node->begin(), node->count(), &(node->bound()));
  }

  // Otherwise, attempt to split.
  else {
    bool can_cut = AttemptSplitting(
                     metric_in, matrix, node, &left, &right,
                     leaf_size, old_from_new);

    if(can_cut) {
      (*current_num_leaf_nodes)++;
      (*num_nodes) = (*num_nodes) + 2;
      SplitGenMetricTree(
        metric_in, matrix, left, leaf_size, max_num_leaf_nodes,
        current_num_leaf_nodes, old_from_new, num_nodes);
      SplitGenMetricTree(
        metric_in, matrix, right, leaf_size, max_num_leaf_nodes,
        current_num_leaf_nodes, old_from_new, num_nodes);
      CombineBounds(metric_in, matrix, node, left, right);
    }
    else {
      MakeLeafMetricTreeNode(
        metric_in, matrix, node->begin(), node->count(), &(node->bound()));
    }
  }

  // Set children information appropriately.
  node->set_children(matrix, left, right);
}
};
};
