/** @file gen_metric_tree_impl.h
 *
 *  Implementation for the regular pointer-style ball-tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

namespace tree_gen_metric_tree_private {

  // This function assumes that we have points embedded in Euclidean
  // space.
  template<typename TBound>
  void MakeLeafMetricTreeNode(const arma::mat& matrix,
			      index_t begin, index_t count, TBound *bounds) {

    bounds->center().zeros();

    index_t end = begin + count;
    for (index_t i = begin; i < end; i++) {
      bounds->center() += matrix.unsafe_col(i);
    }
    bounds->center() /= (double) count;

    double furthest_distance;
    FurthestColumnIndex(bounds->center(), matrix, begin, count,
			&furthest_distance);
    bounds->set_radius(furthest_distance);
  }
  
  template<typename TBound>
  index_t MatrixPartition(arma::mat& matrix, index_t first, index_t count,
			  TBound &left_bound, TBound &right_bound,
			  index_t *old_from_new) {
    
    index_t end = first + count;
    index_t left_count = 0;

    std::vector<bool> left_membership;
    left_membership.reserve(count);
    
    for (index_t left = first; left < end; left++) {

      // Make alias of the current point.
      arma::vec point = matrix.unsafe_col(left);

      // Compute the distances from the two pivots.
      double distance_from_left_pivot =
	LMetric<2>::Distance(point, left_bound.center());
      double distance_from_right_pivot =
	LMetric<2>::Distance(point, right_bound.center());

      // We swap if the point is further away from the left pivot.
      if(distance_from_left_pivot > distance_from_right_pivot) {	
	left_membership[left - first] = false;
      } else {
	left_membership[left - first] = true;
	left_count++;
      }
    }

    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (left_membership[left - first] && (left <= right)) {
        left++;
      }

      while (!left_membership[right - first] && (left <= right)) {
        right--;
      }

      if (left > right) {
        /* left == right + 1 */
        break;
      }

      // Swap the left vector with the right vector.
      matrix.swap_cols(left, right);

      bool tmp = left_membership[left - first];
      left_membership[left - first] = left_membership[right - first];
      left_membership[right - first] = tmp;
      
      if (old_from_new) {
        index_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    
    DEBUG_ASSERT(left == right + 1);

    return left_count;
  }
	
  index_t FurthestColumnIndex(const arma::vec& pivot, const arma::mat& matrix, 
			      index_t begin, index_t count,
			      double *furthest_distance) {
    
    index_t furthest_index = -1;
    index_t end = begin + count;
    *furthest_distance = -1.0;

    for(index_t i = begin; i < end; i++) {
      double distance_between_center_and_point = 
	LMetric<2>::Distance(pivot, matrix.unsafe_col(i));
      
      if((*furthest_distance) < distance_between_center_and_point) {
	*furthest_distance = distance_between_center_and_point;
	furthest_index = i;
      }
    }

    return furthest_index;
  }

  template<typename TMetricTree>
  bool AttemptSplitting(arma::mat& matrix, TMetricTree *node, TMetricTree **left, 
			TMetricTree **right, index_t leaf_size,
			index_t *old_from_new) {

    // Pick a random row.
    index_t random_row = math::RandInt(node->begin(), node->begin() +
				       node->count());
    random_row = node->begin();
    arma::vec random_row_vec = matrix.unsafe_col(random_row);

    // Now figure out the furthest point from the random row picked
    // above.
    double furthest_distance;
    index_t furthest_from_random_row =
      FurthestColumnIndex(random_row_vec, matrix, node->begin(), node->count(),
			  &furthest_distance);
    arma::vec furthest_from_random_row_vec = matrix.unsafe_col(furthest_from_random_row);

    // Then figure out the furthest point from the furthest point.
    double furthest_from_furthest_distance;
    index_t furthest_from_furthest_random_row =
      FurthestColumnIndex(furthest_from_random_row_vec, matrix, node->begin(),
			  node->count(), &furthest_from_furthest_distance);
    arma::vec furthest_from_furthest_random_row_vec =
       matrix.unsafe_col(furthest_from_furthest_random_row);

    if(furthest_from_furthest_distance < DBL_EPSILON) {
      return false;
    } else {
      *left = new TMetricTree();
      *right = new TMetricTree();

      // not necessary, vec::operator=() takes care of resetting the size
//      ((*left)->bound().center()).set_size(matrix.n_rows);
//      ((*right)->bound().center()).set_size(matrix.n_rows);

      ((*left)->bound().center()) = furthest_from_random_row_vec;
      ((*right)->bound().center()) = furthest_from_furthest_random_row_vec;

      index_t left_count = MatrixPartition
	(matrix, node->begin(), node->count(),
	 (*left)->bound(), (*right)->bound(), old_from_new);

      (*left)->Init(node->begin(), left_count);
      (*right)->Init(node->begin() + left_count, node->count() - left_count);
    }

    return true;
  }

  template<typename TMetricTree>
  void CombineBounds(arma::mat& matrix, TMetricTree *node, TMetricTree *left,
		     TMetricTree *right) {
    
    // First clear the internal node center.
    node->bound().center().zeros();

    // Compute the weighted sum of the two pivots
    node->bound().center() += left->count() * left->bound().center();
    node->bound().center() += right->count() * right->bound().center();
    node->bound().center() /= (double) node->count();
    
    double left_max_dist, right_max_dist;
    FurthestColumnIndex(node->bound().center(), matrix, left->begin(), 
			left->count(), &left_max_dist);
    FurthestColumnIndex(node->bound().center(), matrix, right->begin(), 
			right->count(), &right_max_dist);    
    node->bound().set_radius(std::max(left_max_dist, right_max_dist));
  }

  template<typename TMetricTree>
  void SplitGenMetricTree(arma::mat& matrix, TMetricTree *node,
			  index_t leaf_size, index_t *old_from_new) {
    
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
