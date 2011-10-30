// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTCLIN
/**
 * @file gen_cosine_tree.h
 *
 * Tools for cosine-trees.
 *
 * @experimental
 */

#ifndef GEN_COSINE_TREE_H
#define GEN_COSINE_TREE_H

#define NDEBUG

#include <armadillo>
#include <assert.h>

#include "general_spacetree.h"
#include "gen_cosine_tree_impl.h"


/**
 * Regular pointer-style trees (as opposed to THOR trees).
 */
namespace proximity {

  /**
   * Creates a cosine tree from data.
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
  template<typename TCosineTree>
    TCosineTree *MakeGenCosineTree(arma::mat& matrix, size_t leaf_size,
				   arma::Col<size_t> *old_from_new = NULL,
				   arma::Col<size_t> *new_from_old = NULL) {

    TCosineTree *node = new TCosineTree();
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
    node->bound().center() = arma::mean(matrix, 1);

    assert(node->bound().center().n_elem == matrix.n_rows); 

    tree_gen_cosine_tree_private::SplitGenCosineTree<TCosineTree>
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
