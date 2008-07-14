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

#ifndef TREE_KDTREE_MMAP_H
#define TREE_KDTREE_MMAP_H

#include "fastlib/base/base.h"
#include "fastlib/mmanager/memory_manager.h"

#include "spacetree.h"
#include "bounds.h"

#include "fastlib/col/arraylist.h"
#include "fastlib/fx/fx.h"

#include "kdtree_mmap_impl.h"

/**
 * Regular pointer-style trees (as opposed to THOR trees).
 */
namespace tree_mmap {
  /**
   * Creates a KD tree from data, splitting on the midpoint.
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
  TKdTree *MakeKdTreeMidpointSelective(Matrix& matrix, Vector split_dimensions,
      index_t leaf_size,
      GenVector<index_t> *old_from_new = NULL,
      GenVector<index_t> *new_from_old = NULL) {
    TKdTree *node = new TKdTree();
    index_t *old_from_new_ptr;
   
    if (old_from_new) {
      old_from_new->StaticInit(matrix.n_cols());
      
      for (index_t i = 0; i < matrix.n_cols(); i++) {
        (*old_from_new)[i] = i;
      }
      
      old_from_new_ptr = old_from_new->ptr();
    } else {
      old_from_new_ptr = NULL;
    }
      
    node->Init(0, matrix.n_cols());
    node->bound().Init(split_dimensions.length());
    tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
        0, matrix.n_cols(), &node->bound());

    tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions, 
	node, leaf_size, old_from_new_ptr);
    
    if (new_from_old) {
      new_from_old->StaticInit(matrix.n_cols());
      for (index_t i = 0; i < matrix.n_cols(); i++) {
        (*new_from_old)[(*old_from_new)[i]] = i;
      }
    }    
    return node;
  }

  template<typename TKdTree>
    TKdTree *MakeKdTreeMidpoint(Matrix& matrix, index_t leaf_size,
				GenVector<index_t> *old_from_new = NULL,
				GenVector<index_t> *new_from_old = NULL) {  
    Vector split_dimensions;
    split_dimensions.Init(matrix.n_rows());
    int i;
    for (i = 0; i < matrix.n_rows(); i++){
      split_dimensions[i] = i;
    }
    TKdTree *result;
    result = MakeKdTreeMidpointSelective<TKdTree>(matrix, split_dimensions,
		   leaf_size, old_from_new, new_from_old);
    return result;
  }
};

/** Basic KD tree structure. @experimental */
typedef BinarySpaceTree<DHrectBoundMmap<2>, Matrix> BasicKdTreeMmap;

#endif
