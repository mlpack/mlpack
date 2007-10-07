// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file tree/kdtree.h
 *
 * Tools for kd-trees.
 *
 * Eventually we hope to support KD trees with non-L2 (Euclidean)
 * metrics, like Manhattan distance.
 *
 * @experimental
 */

#ifndef TREE_SPILL_KDTREE_H
#define TREE_SPILL_KDTREE_H

#include "general_spacetree.h"

#include "base/common.h"
#include "col/arraylist.h"
#include "fx/fx.h"

#include "spill_kdtree_impl.h"

/**
 * Regular pointer-style trees (as opposed to THOR trees).
 */
namespace proximity {
  /**
   * Creates a spill KD tree from data, splitting on the midpoint.
   *
   * @experimental
   *
   * This requires you to pass in two unitialized ArrayLists which will contain
   * index mappings so you can account for the re-ordering of the matrix.
   * (By unitialized I mean don't call Init on it)
   *
   * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
   * @param leaf_size the maximum points in a leaf
   * @param old_from_new pointer to an unitialized arraylist; it will map
   *        new indices to original
   * @param new_from_old pointer to an unitialized arraylist; it will map
   *        original indexes to new indices
   */
  template<typename TKdTree>
    TKdTree *MakeSpillKdTreeMidpoint(Matrix& matrix, index_t leaf_size,
				     ArrayList<index_t> *old_from_new = NULL,
				     ArrayList<index_t> *new_from_old = NULL) {
    TKdTree *node = new TKdTree();
    index_t *old_from_new_ptr;

    if (old_from_new) {
      old_from_new->Init(matrix.n_cols());
      
      for (index_t i = 0; i < matrix.n_cols(); i++) {
        (*old_from_new)[i] = i;
      }
      
      old_from_new_ptr = old_from_new->begin();
    } else {
      old_from_new_ptr = NULL;
    }
      
    node->Init(0, matrix.n_cols(), matrix.n_cols());
    node->bound().Init(matrix.n_rows());
    tree_spill_kdtree_private::FindBoundFromMatrix(matrix, 0, matrix.n_cols(), 
						   &node->bound());

    tree_spill_kdtree_private::SplitSpillKdTreeMidpoint(matrix, node, 
							leaf_size,
							old_from_new_ptr);
    
    if (new_from_old) {
      new_from_old->Init(matrix.n_cols());
      for (index_t i = 0; i < matrix.n_cols(); i++) {
        (*new_from_old)[(*old_from_new)[i]] = i;
      }
    }
    
    return node;
  }

};

#endif
