/** @file gen_cosine_tree_impl.h
 *
 *  Implementation for the regular pointer-style cosine-tree builder.
 *
 *  @author Parikshit Ram (pram@cc.gatech.edu)
 */
#ifndef GEN_COSINE_TREE_IMPL_H
#define GEN_COSINE_TREE_IMPL_H

#define NDEBUG

#include <assert.h>
#include <mlpack/core.h>
#include <armadillo>
#include "cosine.h"

namespace tree_gen_cosine_tree_private {

  // not compiled
  template<typename TBound>
  void MakeLeafCosineTreeNode(const arma::mat& matrix,
			      size_t begin, size_t count,
			      TBound *bounds) {


    arma::vec cosine_vec(count);
    
    for (size_t i = 0; i < count; i++)
      cosine_vec(i) = Cosine::Evaluate(bounds->center(), 
				       matrix.unsafe_col(i + begin));
    
    bounds->set_radius(arma::min(cosine_vec), arma::max(cosine_vec));
  }
  

  // not compiled
  template<typename TCosineTree>
  bool AttemptSplitting(arma::mat& matrix,
			TCosineTree *node,
			TCosineTree **left, 
			TCosineTree **right, 
			size_t leaf_size,
			size_t *old_from_new) {

    // obtain the list of cosine values to all the points
    // in the set
    arma::vec cosine_vec(node->count());

    for (size_t i = 0; i < node->count(); i++)
      cosine_vec(i)
	= Cosine::Evaluate(node->bound().center(),
			   matrix.unsafe_col(i + node->begin()));

    if(arma::max(cosine_vec) - arma::min(cosine_vec) < DBL_EPSILON) {
      // everything in a really tight narrow co-axial cone-ring
      return false;
    } else {
      *left = new TCosineTree();
      *right = new TCosineTree();

      ((*left)->bound().center()) = node->bound().center();
      ((*right)->bound().center()) = node->bound().center();

      node->bound().set_radius(arma::min(cosine_vec), arma::max(cosine_vec));

//       printf("%lg, %lg\n", node->bound().rad_min(), node->bound().rad_max());

      size_t first = node->begin();
      size_t end = first + node->count();
      size_t left_count = 0;

      double median_cosine_value = arma::median(cosine_vec);

      std::vector<bool> left_membership;
      left_membership.reserve(node->count());
    
      for (size_t left_ind = first; left_ind < end; left_ind++) {
	  
	if(cosine_vec(left_ind - first) < median_cosine_value) {	
	  // the outer ring
	  left_membership[left_ind - first] = false;
	} else {
	  // the inner ring
	  left_membership[left_ind - first] = true;
	  left_count++;
	}
      }

      size_t left_ind = first;
      size_t right_ind = end - 1;
    
      /* At any point:
       *
       *   everything < left_ind is correct
       *   everything > right_ind is correct
       */
      for (;;) {
	while (left_membership[left_ind - first] && (left_ind <= right_ind)) {
	  left_ind++;
	}

	while (!left_membership[right_ind - first] && (left_ind <= right_ind)) {
	  right_ind--;
	}

	if (left_ind > right_ind) {
	  /* left == right_ind + 1 */
	  break;
	}

	// Swap the left vector with the right_ind vector.
	matrix.swap_cols(left_ind, right_ind);

	bool tmp = left_membership[left_ind - first];
	left_membership[left_ind - first] = left_membership[right_ind - first];
	left_membership[right_ind - first] = tmp;
      
	if (old_from_new) {
	  size_t t = old_from_new[left_ind];
	  old_from_new[left_ind] = old_from_new[right_ind];
	  old_from_new[right_ind] = t;
	}
      
	assert(left_ind <= right_ind);
	right_ind--;
      }
    
      assert(left_ind == right_ind + 1);

      (*left)->Init(node->begin(), left_count);
      (*right)->Init(node->begin() + left_count, node->count() - left_count);
	
      return true;    
    }
  }


  // not compiled
  template<typename TCosineTree>
  void SplitGenCosineTree(arma::mat& matrix, TCosineTree *node,
			  size_t leaf_size, size_t *old_from_new) {
    
    TCosineTree *left = NULL;
    TCosineTree *right = NULL;

    // If the node is just too small, then do not split.
    if(node->count() < leaf_size) {
      MakeLeafCosineTreeNode(matrix, node->begin(), node->count(),
			     &(node->bound()));
    }
    
    // Otherwise, attempt to split.
    else {
      bool can_cut = AttemptSplitting(matrix, node, &left, &right, 
				      leaf_size, old_from_new);
      
      if(can_cut) {
	SplitGenCosineTree(matrix, left, leaf_size, old_from_new);
	SplitGenCosineTree(matrix, right, leaf_size, old_from_new);
      }
      else {
	MakeLeafCosineTreeNode(matrix, node->begin(),
			       node->count(), &(node->bound()));
      }
    }
    
    // Set children information appropriately.
    node->set_children(matrix, left, right);
  }
};

#endif
