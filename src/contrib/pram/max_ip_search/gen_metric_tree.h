// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTCLIN
/**
 * @file tree/kdtree.h
 *
 * Tools for ball-trees.
 *
 * Eventually we hope to support KD trees with non-L2 (Euclidean)
 * metrics, like Manhattan distance.
 *
 * @experimental
 */

#ifndef GEN_METRIC_TREE_H
#define GEN_METRIC_TREE_H

#define NDEBUG

#include "general_spacetree.h"
#include "gen_metric_tree_impl.h"

/**
 * Regular pointer-style trees (as opposed to THOR trees).
 */
namespace proximity {

  /**
   * Creates a ball tree from data.
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
  template<typename TMetricTree>
  TMetricTree *MakeGenMetricTree(arma::mat& matrix, size_t leaf_size,
				 arma::Col<size_t> *old_from_new = NULL,
				 arma::Col<size_t> *new_from_old = NULL) {

    TMetricTree *node = new TMetricTree();
    size_t *old_from_new_ptr;

    if (old_from_new) {
      old_from_new->set_size(matrix.n_cols);
      
      for (size_t i = 0; i < matrix.n_cols; i++) {
        (*old_from_new)[i] = i;
      }
      
      old_from_new_ptr = old_from_new->memptr();
    } else {
      old_from_new_ptr = NULL;
    }

    node->Init(0, matrix.n_cols);
    node->bound().center().set_size(matrix.n_rows);
    tree_gen_metric_tree_private::SplitGenMetricTree<TMetricTree>
      (matrix, node, leaf_size, old_from_new_ptr);
    
    if (new_from_old) {
      new_from_old->set_size(matrix.n_cols);
      for (size_t i = 0; i < matrix.n_cols; i++) {
        (*new_from_old)[(*old_from_new)[i]] = i;
      }
    }
    
    return node;
  }

};

#endif
