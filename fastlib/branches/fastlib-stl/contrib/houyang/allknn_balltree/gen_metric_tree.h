// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 *
 * Tools for learning ball-trees.
 *
 * Eventually we hope to support KD trees with non-L2 (Euclidean)
 * metrics, like Manhattan distance.
 *
 * @experimental
 */

#ifndef TREE_GEN_METRIC_TREE_H
#define TREE_GEN_METRIC_TREE_H

#include "general_spacetree.h"

#include "fastlib/base/common.h"
#include "fastlib/col/arraylist.h"
#include "fastlib/fx/fx.h"

#include "gen_metric_tree_impl.h"

#include "allknn_balltree.h"

/**
 * Regular pointer-style trees (as opposed to THOR trees).
 */
namespace learntrees {

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

  /**
   * Make a traditional ball tree
   */
  template<typename TMetricTree>
  TMetricTree *MakeGenMetricTree(Matrix& matrix, index_t leaf_size,
				 ArrayList<index_t> *old_from_new = NULL,
				 ArrayList<index_t> *new_from_old = NULL) {
    TMetricTree *node = new TMetricTree();
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

    node->Init(0, matrix.n_cols(), matrix.n_rows());
    node->bound().center().Init(matrix.n_rows());
    learntrees_private::SplitGenMetricTree<TMetricTree>
      (matrix, node, leaf_size, old_from_new_ptr, NULL);
    
    if (new_from_old) {
      new_from_old->Init(matrix.n_cols());
      for (index_t i = 0; i < matrix.n_cols(); i++) {
        (*new_from_old)[(*old_from_new)[i]] = i;
      }
    }
    
    return node;
  }

  /**
   * Learn a ball tree
   */
  template<typename TMetricTree>
  TMetricTree *LearnGenMetricTree(Matrix& ref_matrix, index_t leaf_size, index_t knns,
				  const Vector& D, const Matrix& Adj, const Matrix& Aff,
				  ArrayList<index_t> *old_from_new = NULL,
				  ArrayList<index_t> *new_from_old = NULL) {
    TMetricTree *node = new TMetricTree();
    index_t *old_from_new_ptr, *new_from_old_ptr;

    if (old_from_new) {
      old_from_new->Init(ref_matrix.n_cols());
      for (index_t i = 0; i < ref_matrix.n_cols(); i++) {
        (*old_from_new)[i] = i;
      }
      old_from_new_ptr = old_from_new->begin();
    }
    else {
      old_from_new_ptr = NULL;
    }

    if (new_from_old) {
      new_from_old->Init(ref_matrix.n_cols());
      for (index_t i = 0; i < ref_matrix.n_cols(); i++) {
	(*new_from_old)[i] = i;
      }
      new_from_old_ptr = new_from_old->begin();
    }
    else {
      new_from_old_ptr = NULL;
    }
    
    //node->LearnInit(0, ref_matrix.n_cols(), 0, ref_matrix.n_rows());
    node->Init(0, ref_matrix.n_cols(), ref_matrix.n_rows());

    node->bound().center().Init(ref_matrix.n_rows());
    learntrees_private::LearnSplitGenMetricTree<TMetricTree>
      (ref_matrix, node, leaf_size, old_from_new_ptr, new_from_old_ptr, knns, D, Adj, Aff);
    
    // Done in learntrees_private::LearnMatrixPartition()
    //if (new_from_old) {
    //  new_from_old->Init(ref_matrix.n_cols());
    //  for (index_t i = 0; i < ref_matrix.n_cols(); i++) {
    //    (*new_from_old)[(*old_from_new)[i]] = i;
    //  }
    //}
    return node;
  }


};

#endif
