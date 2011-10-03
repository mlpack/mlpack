/** @file gen_metric_tree_impl.h
 *
 *  Implementation for the regular pointer-style ball-tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "fastlib/fastlib_int.h"

namespace tree_gen_metric_tree_private {

  // This function assumes that we have points embedded in Euclidean
  // space.
  template<typename TBound>
  void MakeLeafMetricTreeNode(const Matrix& matrix,
			      size_t begin, size_t count, TBound *bounds) {

    bounds->center().SetZero();

    size_t end = begin + count;
    for (size_t i = begin; i < end; i++) {
      Vector col;      
      matrix.MakeColumnVector(i, &col);
      la::AddTo(col, &(bounds->center()));
    }
    la::Scale(1.0 / ((double) count), &(bounds->center()));

    double furthest_distance;
    FurthestColumnIndex(bounds->center(), matrix, begin, count,
			&furthest_distance);
    bounds->set_radius(furthest_distance);
  }
  
  template<typename TBound>
  size_t MatrixPartition(Matrix& matrix, size_t first, size_t count,
			  TBound &left_bound, TBound &right_bound,
			  size_t *old_from_new) {
    
    size_t end = first + count;
    size_t left_count = 0;

    ArrayList<bool> left_membership;
    left_membership.Init(count);
    
    for (size_t left = first; left < end; left++) {

      // Make alias of the current point.
      Vector point;
      matrix.MakeColumnVector(left, &point);

      // Compute the distances from the two pivots.
      double distance_from_left_pivot =
	LMetric<2>::Distance(point, left_bound.center());
      double distance_from_right_pivot =
	LMetric<2>::Distance(point, right_bound.center());

      // We swap if the point is further away from the left pivot.
      if(distance_from_left_pivot > distance_from_right_pivot) {	
	left_membership[left - first] = false;
      }
      else {
	left_membership[left - first] = true;
	left_count++;
      }
    }

    size_t left = first;
    size_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (left_membership[left - first] && likely(left <= right)) {
        left++;
      }

      while (!left_membership[right - first] && likely(left <= right)) {
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      Vector left_vector;
      Vector right_vector;

      matrix.MakeColumnVector(left, &left_vector);
      matrix.MakeColumnVector(right, &right_vector);

      // Swap the left vector with the right vector.
      left_vector.SwapValues(&right_vector);
      bool tmp = left_membership[left - first];
      left_membership[left - first] = left_membership[right - first];
      left_membership[right - first] = tmp;
      
      if (old_from_new) {
        size_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    
    DEBUG_ASSERT(left == right + 1);

    return left_count;
  }
	
  size_t FurthestColumnIndex(const Vector &pivot, const Matrix &matrix, 
			      size_t begin, size_t count,
			      double *furthest_distance) {
    
    size_t furthest_index = -1;
    size_t end = begin + count;
    *furthest_distance = -1.0;

    for(size_t i = begin; i < end; i++) {
      Vector point;
      matrix.MakeColumnVector(i, &point);
      double distance_between_center_and_point = 
	LMetric<2>::Distance(pivot, point);
      
      if((*furthest_distance) < distance_between_center_and_point) {
	*furthest_distance = distance_between_center_and_point;
	furthest_index = i;
      }
    }

    return furthest_index;
  }

  template<typename TMetricTree>
  bool AttemptSplitting(Matrix& matrix, TMetricTree *node, TMetricTree **left, 
			TMetricTree **right, size_t leaf_size,
			size_t *old_from_new) {

    // Pick a random row.
    size_t random_row = math::RandInt(node->begin(), node->begin() +
				       node->count());
    random_row = node->begin();
    Vector random_row_vec;
    matrix.MakeColumnVector(random_row, &random_row_vec);

    // Now figure out the furthest point from the random row picked
    // above.
    double furthest_distance;
    size_t furthest_from_random_row =
      FurthestColumnIndex(random_row_vec, matrix, node->begin(), node->count(),
			  &furthest_distance);
    Vector furthest_from_random_row_vec;
    matrix.MakeColumnVector(furthest_from_random_row,
			    &furthest_from_random_row_vec);

    // Then figure out the furthest point from the furthest point.
    double furthest_from_furthest_distance;
    size_t furthest_from_furthest_random_row =
      FurthestColumnIndex(furthest_from_random_row_vec, matrix, node->begin(),
			  node->count(), &furthest_from_furthest_distance);
    Vector furthest_from_furthest_random_row_vec;
    matrix.MakeColumnVector(furthest_from_furthest_random_row,
			    &furthest_from_furthest_random_row_vec);

    if(furthest_from_furthest_distance < DBL_EPSILON) {
      return false;
    }
    else {
      *left = new TMetricTree();
      *right = new TMetricTree();

      ((*left)->bound().center()).Init(matrix.n_rows());
      ((*right)->bound().center()).Init(matrix.n_rows());

      ((*left)->bound().center()).CopyValues(furthest_from_random_row_vec);
      ((*right)->bound().center()).CopyValues
	(furthest_from_furthest_random_row_vec);

      size_t left_count = MatrixPartition
	(matrix, node->begin(), node->count(),
	 (*left)->bound(), (*right)->bound(), old_from_new);

      (*left)->Init(node->begin(), left_count);
      (*right)->Init(node->begin() + left_count, node->count() - left_count);
    }

    return true;
  }

  template<typename TMetricTree>
  void CombineBounds(Matrix &matrix, TMetricTree *node, TMetricTree *left,
		     TMetricTree *right) {
    
    // First clear the internal node center.
    node->bound().center().SetZero();

    // Compute the weighted sum of the two pivots
    la::AddExpert(left->count(), left->bound().center(),
		  &(node->bound().center()));
    la::AddExpert(right->count(), right->bound().center(),
		  &(node->bound().center()));
    la::Scale(1.0 / ((double) node->count()), &(node->bound().center()));
    
    double left_max_dist, right_max_dist;
    FurthestColumnIndex(node->bound().center(), matrix, left->begin(), 
			left->count(), &left_max_dist);
    FurthestColumnIndex(node->bound().center(), matrix, right->begin(), 
			right->count(), &right_max_dist);    
    node->bound().set_radius(std::max(left_max_dist, right_max_dist));
  }

  template<typename TMetricTree>
  void SplitGenMetricTree(Matrix& matrix, TMetricTree *node,
			  size_t leaf_size, size_t *old_from_new) {
    
    TMetricTree *left = NULL;
    TMetricTree *right = NULL;

    // If the node is just too small, then do not split.
    if(node->count() < leaf_size) {
      MakeLeafMetricTreeNode(matrix, node->begin(), node->count(),
			     &(node->bound()));
    }
    
    // Otherwise, attempt to split.
    else {
      bool can_cut = AttemptSplitting(matrix, node, &left, &right, leaf_size,
				      old_from_new);
      
      if(can_cut) {
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
