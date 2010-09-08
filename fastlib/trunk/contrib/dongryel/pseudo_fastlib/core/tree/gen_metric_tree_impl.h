/** @file gen_metric_tree_impl.h
 *
 *  Implementation for the regular pointer-style ball-tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <armadillo>
#include <deque>
#include "core/metric_kernels/lmetric.h"

namespace core {
namespace tree_private {

// This function assumes that we have points embedded in Euclidean
// space.
template<typename TBound>
void MakeLeafMetricTreeNode(
  const arma::mat& matrix,
  int begin, int count, TBound *bounds) {

  bounds->center().fill(0.0);

  int end = begin + count;
  for (int i = begin; i < end; i++) {
    arma::vec col = matrix.col(i);
    bounds->center() += col;
  }
  bounds->center() = bounds->center() / ((double) count);

  double furthest_distance;
  FurthestColumnIndex(
    bounds->center(), matrix, begin, count, &furthest_distance);
  bounds->set_radius(furthest_distance);
}

template<typename TBound>
int MatrixPartition(
  arma::mat& matrix, int first, int count,
  TBound &left_bound, TBound &right_bound, std::vector<int> *old_from_new) {

  int end = first + count;
  int left_count = 0;

  std::deque<bool> left_membership;
  left_membership.resize(count);

  for (int left = first; left < end; left++) {

    // Make alias of the current point.
    arma::vec point = matrix.col(left);

    // Compute the distances from the two pivots.
    double distance_from_left_pivot =
      core::metric_kernels::LMetric<2>::Distance(point, left_bound.center());
    double distance_from_right_pivot =
      core::metric_kernels::LMetric<2>::Distance(point, right_bound.center());

    // We swap if the point is further away from the left pivot.
    if (distance_from_left_pivot > distance_from_right_pivot) {
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
  for (;;) {
    while (left_membership[left - first] && left <= right) {
      left++;
    }

    while (!left_membership[right - first] && left <= right) {
      right--;
    }

    if (left > right) {

      // left == right + 1
      break;
    }

    // Swap the left vector with the right vector.
    matrix.swap_cols(left, right);
    bool tmp = left_membership[left - first];
    left_membership[left - first] = left_membership[right - first];
    left_membership[right - first] = tmp;

    if (old_from_new) {
      int t = (*old_from_new)[left];
      (*old_from_new)[left] = (*old_from_new)[right];
      (*old_from_new)[right] = t;
    }
    right--;
  }

  return left_count;
}

int FurthestColumnIndex(
  const arma::vec &pivot, const arma::mat &matrix,
  int begin, int count,
  double *furthest_distance) {

  int furthest_index = -1;
  int end = begin + count;
  *furthest_distance = -1.0;

  for (int i = begin; i < end; i++) {
    arma::vec point = matrix.col(i);
    double distance_between_center_and_point =
      core::metric_kernels::LMetric<2>::Distance(pivot, point);

    if ((*furthest_distance) < distance_between_center_and_point) {
      *furthest_distance = distance_between_center_and_point;
      furthest_index = i;
    }
  }

  return furthest_index;
}

template<typename TMetricTree>
bool AttemptSplitting(
  arma::mat& matrix, TMetricTree *node, TMetricTree **left,
  TMetricTree **right, int leaf_size,
  std::vector<int> *old_from_new) {

  // Pick a random row.
  int random_row = math::RandInt(node->begin(), node->begin() +
                                 node->count());
  random_row = node->begin();
  arma::vec random_row_vec = matrix.col(random_row);

  // Now figure out the furthest point from the random row picked
  // above.
  double furthest_distance;
  int furthest_from_random_row =
    FurthestColumnIndex(random_row_vec, matrix, node->begin(), node->count(),
                        &furthest_distance);
  arma::vec furthest_from_random_row_vec = matrix.col(furthest_from_random_row);

  // Then figure out the furthest point from the furthest point.
  double furthest_from_furthest_distance;
  int furthest_from_furthest_random_row =
    FurthestColumnIndex(furthest_from_random_row_vec, matrix, node->begin(),
                        node->count(), &furthest_from_furthest_distance);
  arma::vec furthest_from_furthest_random_row_vec =
    matrix.col(furthest_from_furthest_random_row);

  if (furthest_from_furthest_distance <
      std::numeric_limits<double>::epsilon()) {
    return false;
  }
  else {
    *left = new TMetricTree();
    *right = new TMetricTree();

    ((*left)->bound().center()).set_size(matrix.n_rows);
    ((*right)->bound().center()).set_size(matrix.n_rows);

    ((*left)->bound().center()) = furthest_from_random_row_vec;
    ((*right)->bound().center()) = furthest_from_furthest_random_row_vec;

    int left_count = MatrixPartition(
                       matrix, node->begin(), node->count(),
                       (*left)->bound(), (*right)->bound(), old_from_new);

    (*left)->Init(node->begin(), left_count);
    (*right)->Init(node->begin() + left_count, node->count() - left_count);
  }

  return true;
}

template<typename TMetricTree>
void CombineBounds(
  arma::mat &matrix, TMetricTree *node, TMetricTree *left, TMetricTree *right) {

  // Compute the weighted sum of the two pivots
  node->bound().center() = left->count() * left->bound().center();
  node->bound().center() = node->bound().center() +
                           right->count() * right->bound().center();
  node->bound().center() = node->bound().center() / ((double) node->count());

  double left_max_dist, right_max_dist;
  FurthestColumnIndex(
    node->bound().center(), matrix, left->begin(),
    left->count(), &left_max_dist);
  FurthestColumnIndex(
    node->bound().center(), matrix, right->begin(),
    right->count(), &right_max_dist);
  node->bound().set_radius(std::max(left_max_dist, right_max_dist));
}

template<typename TMetricTree>
void SplitGenMetricTree(
  arma::mat& matrix, TMetricTree *node,
  int leaf_size, std::vector<int> *old_from_new) {

  TMetricTree *left = NULL;
  TMetricTree *right = NULL;

  // If the node is just too small, then do not split.
  if (node->count() < leaf_size) {
    MakeLeafMetricTreeNode(
      matrix, node->begin(), node->count(), &(node->bound()));
  }

  // Otherwise, attempt to split.
  else {
    bool can_cut = AttemptSplitting(
                     matrix, node, &left, &right, leaf_size, old_from_new);

    if (can_cut) {
      SplitGenMetricTree(matrix, left, leaf_size, old_from_new);
      SplitGenMetricTree(matrix, right, leaf_size, old_from_new);
      CombineBounds(matrix, node, left, right);
    }
    else {
      MakeLeafMetricTreeNode(matrix, node->begin(), node->count(),
                             &(node->bound()));
    }
  }

  // Set children information appropriately.
  node->set_children(matrix, left, right);
}
};
};
