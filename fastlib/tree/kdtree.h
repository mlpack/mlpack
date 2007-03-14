// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file kdtree.h
 *
 * Tools for kd-trees.
 *
 * Eventually we hope to support KD trees with non-L2 (Euclidean)
 * metrics, like Manhattan distance.
 */

#ifndef TREE_KDTREE_H
#define TREE_KDTREE_H

#include "spacetree.h"
#include "bounds.h"

#include "base/common.h"
#include "col/arraylist.h"
#include "file/serialize.h"

/* Implementation */

namespace tree_kdtree_private {

  template<typename TBound>
  void FindBoundFromMatrix(const Matrix& matrix,
      index_t first, index_t count, TBound *bounds) {
    index_t end = first + count;
    for (index_t i = first; i < end; i++) {
      Vector col;

      matrix.MakeColumnVector(i, &col);
      bounds->Update(col);
    }
  }

  template<typename TBound>
  index_t MatrixPartition(
      Matrix& matrix, index_t dim, double splitvalue,
      index_t first, index_t count,
      TBound* left_bound, TBound* right_bound,
      index_t *old_from_new) {
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix.get(dim, left) < splitvalue && likely(left <= right)) {
        Vector left_vector;
        matrix.MakeColumnVector(left, &left_vector);
        left_bound->Update(left_vector);
        left++;
      }

      while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
        Vector right_vector;
        matrix.MakeColumnVector(right, &right_vector);
        right_bound->Update(right_vector);
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

      left_vector.SwapValues(&right_vector);

      left_bound->Update(left_vector);
      right_bound->Update(right_vector);
      
      if (old_from_new) {
        index_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }

      DEBUG_ASSERT(left <= right);
      right--;
      
      // this conditional is always true, I belueve
      //if (likely(left <= right)) {
      //  right--;
      //}
    }

    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename TKdTree>
  void SplitKdTreeMidpoint(Matrix& matrix,
      TKdTree *node, index_t leaf_size, index_t *old_from_new) {
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    //FindBoundFromMatrix(matrix, node->begin(), node->count(),
    //    &node->bound());

    if (node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      double max_width = -1;

      for (index_t d = 0; d < matrix.n_rows(); d++) {
        double w = node->bound().get(d).width();

        if (unlikely(w > max_width)) {
          max_width = w;
          split_dim = d;
        }
      }

      double split_val = node->bound().get(split_dim).mid();

      if (max_width == 0) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } else {
        left = new TKdTree();
        left->bound().Init(matrix.n_rows());

        right = new TKdTree();
        right->bound().Init(matrix.n_rows());

        index_t split_col = MatrixPartition(matrix, split_dim, split_val,
            node->begin(), node->count(),
            &left->bound(), &right->bound(),
            old_from_new);
        
        DEBUG_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
            node->begin(), split_col,
            node->begin() + node->count(), split_dim, split_val,
            node->bound().get(split_dim).lo,
            node->bound().get(split_dim).hi);

        left->Init(node->begin(), split_col - node->begin());
        right->Init(split_col, node->begin() + node->count() - split_col);

        // This should never happen if max_width > 0
        DEBUG_ASSERT(left->count() != 0 && right->count() != 0);

        SplitKdTreeMidpoint(matrix, left, leaf_size, old_from_new);
        SplitKdTreeMidpoint(matrix, right, leaf_size, old_from_new);
      }
    }

    node->set_children(matrix, left, right);
  }
};

namespace tree {
  /**
   * Creates a KD tree from data, splitting on the midpoint.
   *
   * This requires you to pass in two unitialized ArrayLists which will contain
   * index mappings so you can account for the re-ordering of the matrix.
   * (By unitialized I mean don't call Init on it)
   *
   * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
   * @param old_from_new pointer to an unitialized arraylist; it will map
   *        original indexes to new indices
   * @param old_from_new pointer to an unitialized arraylist; it will map
   *        new indices to original
   */
  template<typename TKdTree>
  TKdTree *MakeKdTreeMidpoint(Matrix& matrix, index_t leaf_size,
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
      
    node->Init(0, matrix.n_cols());
    node->bound().Init(matrix.n_rows());
    tree_kdtree_private::FindBoundFromMatrix(matrix,
        0, matrix.n_cols(), &node->bound());

    tree_kdtree_private::SplitKdTreeMidpoint(matrix, node, leaf_size,
        old_from_new_ptr);
    
    if (new_from_old) {
      new_from_old->Init(matrix.n_cols());
      for (index_t i = 0; i < matrix.n_cols(); i++) {
        (*new_from_old)[(*old_from_new)[i]] = i;
      }
    }
    
    return node;
  }

  // TODO: Perhaps move this into a "util.h" file
  
  template<typename TKdTree, typename Serializer>
  void SerializeKdTree(const TKdTree *tree,
      const Matrix& matrix,
      const ArrayList<index_t>& old_from_new,
      Serializer *s) {
    s->PutMagic(file::CreateMagic("kdtree"));
    tree->SerializeAll(matrix, s);
    old_from_new.Serialize(s);
  }

  template<typename TKdTree, typename Deserializer>
  void DeserializeKdTree(TKdTree *uninit_tree,
      Matrix* uninit_matrix,
      ArrayList<index_t>* uninit_old_from_new,
      Deserializer *s) {
    s->AssertMagic(file::CreateMagic("kdtree"));
    uninit_tree->DeserializeAll(uninit_matrix, s);
    if (uninit_old_from_new) {
      uninit_old_from_new->Deserialize(s);
    }
  }

  template<typename TKdTree>
  void ReadKdTreeFromFile(const char *fname,
      TKdTree *uninit_tree,
      Matrix* uninit_matrix,
      ArrayList<index_t>* uninit_old_from_new) {
    NativeFileDeserializer ds;
    
    ASSERT_PASS(ds.Init(fname));
    DeserializeKdTree(uninit_tree, uninit_matrix, uninit_old_from_new,
        &ds);
  }
};

typedef BinarySpaceTree<DHrectBound, Matrix> BasicKdTree;

#endif
