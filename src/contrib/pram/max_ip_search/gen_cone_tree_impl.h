/** @file gen_cone_tree_impl.h
 *
 *  Implementation for the regular pointer-style cone-tree builder.
 *
 *  @author Parikshit Ram (pram@cc.gatech.edu)
 */
#ifndef GEN_CONE_TREE_IMPL_H
#define GEN_CONE_TREE_IMPL_H

#include <assert.h>
#include <mlpack/core.h>
#include <armadillo>
#include "cosine.h"

namespace tree_gen_cone_tree_private {

  size_t FurthestColumnIndex(const arma::vec& pivot,
			     const arma::mat& matrix, 
			     size_t begin, size_t count,
			     double *furthest_cosine);
    
  // This function assumes that we have points embedded in Euclidean
  // space. The representative point (center) is chosen as the mean 
  // of the all the vectors in this set.

  template<typename TBound>
    void MakeLeafConeTreeNode(const arma::mat& matrix,
			      size_t begin, size_t count,
			      TBound *bounds) {

    bounds->center().zeros();

    size_t end = begin + count;
    for (size_t i = begin; i < end; i++) {
      bounds->center() 
	+= (matrix.unsafe_col(i)
 	    / arma::norm(matrix.unsafe_col(i), 2));
    }
    bounds->center() /= (double) count;

    double furthest_cosine;
    FurthestColumnIndex(bounds->center(), matrix, begin, count,
			&furthest_cosine);
    bounds->set_radius(furthest_cosine);
  }
  

  template<typename TBound>
    size_t MatrixPartition(arma::mat& matrix, size_t first, size_t count,
			   TBound &left_bound, TBound &right_bound,
			   size_t *old_from_new) {
    
    size_t end = first + count;
    size_t left_count = 0;

    std::vector<bool> left_membership;
    left_membership.reserve(count);
    
    for (size_t left = first; left < end; left++) {

      // Make alias of the current point.
      arma::vec point = matrix.unsafe_col(left);

      // Compute the cosines from the two pivots.
      double cosine_from_left_pivot =
	Cosine::Evaluate(point, left_bound.center());
      double cosine_from_right_pivot =
	Cosine::Evaluate(point, right_bound.center());

      // We swap if the point is more angled away from the left pivot.
      if(cosine_from_left_pivot < cosine_from_right_pivot) {	
	left_membership[left - first] = false;
      } else {
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
        size_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }
      
      assert(left <= right);
      right--;
    }
    
    assert(left == right + 1);

    return left_count;
  }
	

  template<typename TConeTree>
    bool AttemptSplitting(arma::mat& matrix,
			  TConeTree *node,
			  TConeTree **left, 
			  TConeTree **right, 
			  size_t leaf_size,
			  size_t *old_from_new) {

    // Pick a random row.
    size_t random_row 
      = math::RandInt(node->begin(),
		      node->begin() + node->count());

    arma::vec random_row_vec = matrix.unsafe_col(random_row);

    // Now figure out the furthest point from the random row picked
    // above.
    double furthest_cosine;
    size_t furthest_from_random_row =
      FurthestColumnIndex(random_row_vec, matrix, node->begin(), node->count(),
			  &furthest_cosine);
    arma::vec furthest_from_random_row_vec = matrix.unsafe_col(furthest_from_random_row);

    // Then figure out the furthest point from the furthest point.
    double furthest_from_furthest_cosine;
    size_t furthest_from_furthest_random_row =
      FurthestColumnIndex(furthest_from_random_row_vec, matrix, node->begin(),
			  node->count(), &furthest_from_furthest_cosine);
    arma::vec furthest_from_furthest_random_row_vec =
      matrix.unsafe_col(furthest_from_furthest_random_row);

    if(furthest_from_furthest_cosine > (1.0 - DBL_EPSILON)) {
      // everything in a really tight narrow cone
      return false;
    } else {
      *left = new TConeTree();
      *right = new TConeTree();

      ((*left)->bound().center()) = furthest_from_random_row_vec;
      ((*right)->bound().center()) = furthest_from_furthest_random_row_vec;

      size_t left_count
	= MatrixPartition(matrix, node->begin(), node->count(),
			  (*left)->bound(), (*right)->bound(),
			  old_from_new);

      (*left)->Init(node->begin(), left_count);
      (*right)->Init(node->begin() + left_count, node->count() - left_count);
    }

    return true;
  }

  template<typename TConeTree>
    void CombineBounds(arma::mat& matrix, TConeTree *node,
		       TConeTree *left, TConeTree *right) {
    
    // First clear the internal node center.
    node->bound().center().zeros();

    // Compute the weighted sum of the two pivots
    node->bound().center() += left->count() * left->bound().center();
    node->bound().center() += right->count() * right->bound().center();
    node->bound().center() /= (double) node->count();

    double left_min_cosine, right_min_cosine;
    FurthestColumnIndex(node->bound().center(), matrix, left->begin(), 
			left->count(), &left_min_cosine);
    FurthestColumnIndex(node->bound().center(), matrix, right->begin(), 
			right->count(), &right_min_cosine);    
    node->bound().set_radius(std::min(left_min_cosine, right_min_cosine));
  }

  // fixed
  template<typename TConeTree>
    void SplitGenConeTree(arma::mat& matrix, TConeTree *node,
			  size_t leaf_size, size_t *old_from_new) {
    
    TConeTree *left = NULL;
    TConeTree *right = NULL;

    // If the node is just too small, then do not split.
    if(node->count() < leaf_size) {
      MakeLeafConeTreeNode(matrix, node->begin(), node->count(),
			   &(node->bound()));
    }
    
    // Otherwise, attempt to split.
    else {
      bool can_cut = AttemptSplitting(matrix, node, &left, &right, 
				      leaf_size, old_from_new);
      
      if(can_cut) {
	SplitGenConeTree(matrix, left, leaf_size, old_from_new);
	SplitGenConeTree(matrix, right, leaf_size, old_from_new);
	CombineBounds(matrix, node, left, right);
      }
      else {
	MakeLeafConeTreeNode(matrix, node->begin(),
			     node->count(), &(node->bound()));
      }
    }
    
    // Set children information appropriately.
    node->set_children(matrix, left, right);
  }
};

#endif
