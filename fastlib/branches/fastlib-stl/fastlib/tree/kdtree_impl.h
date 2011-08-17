/* Implementation for the regular pointer-style kd-tree builder. */

#ifndef TREE_KDTREE_IMPL_H
#define TREE_KDTREE_IMPL_H

#include "../base/arma_compat.h"

namespace mlpack {
namespace tree_kdtree_private {
  void MakeBoundVector(const arma::vec& point,
                       const arma::uvec& bound_dimensions,
                       arma::vec& bound_vector);

  template<typename TBound, typename T>
  void SelectFindBoundFromMatrix(const arma::Mat<T>& matrix,
                                 const arma::uvec& split_dimensions,
                                 index_t first,
                                 index_t count, 
                                 TBound *bounds) {
    index_t end = first + count;
    for(index_t i = first; i < end; i++) {
      if (split_dimensions.n_elem == matrix.n_rows){
	*bounds |= matrix.col(i);
      } else {
        arma::vec sub_col(split_dimensions.n_elem);
	MakeBoundVector(matrix.col(i), split_dimensions, sub_col);
	*bounds |= sub_col;
      }          
    }
  }

  template<typename TBound, typename T>
  void FindBoundFromMatrix(const arma::Mat<T>& matrix,
                           index_t first,
                           index_t count,
                           TBound *bounds) {
    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;
    
    SelectFindBoundFromMatrix(matrix, split_dimensions, first, count, bounds);
  }

  template<typename TBound>
  index_t MatrixPartition(arma::mat& matrix,
                          index_t dim,
                          double splitvalue,
                          index_t first,
                          index_t count,
                          TBound& left_bound,
                          TBound& right_bound,
                          arma::Col<index_t>& old_from_new) {
    // we will split dimensions in a very simple order: first dim first
    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    index_t split_point = SelectMatrixPartition(matrix, split_dimensions, 
        dim, splitvalue, first, count, left_bound, right_bound, old_from_new);
    return split_point;
  }
  
  template<typename TBound>
  index_t MatrixPartition(arma::mat& matrix,
                          index_t dim,
                          double splitvalue,
                          index_t first,
                          index_t count,
                          TBound& left_bound,
                          TBound& right_bound) {
    // we will split dimensions in a very simple order: first dim first
    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    index_t split_point = SelectMatrixPartition(matrix, split_dimensions, 
        dim, splitvalue, first, count, left_bound, right_bound);
    return split_point;
  }
  
  template<typename TBound, typename T>
  index_t SelectMatrixPartition(arma::Mat<T>& matrix, 
                                const arma::uvec& split_dimensions,
                                index_t dim,
                                double splitvalue,
                                index_t first,
                                index_t count,
                                TBound& left_bound,
                                TBound& right_bound) {
    
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix(dim, left) < splitvalue && likely(left <= right)) {
	arma::vec left_vector = matrix.col(left);
        if (split_dimensions.n_elem == matrix.n_rows) {
	  left_bound |= left_vector;
	} else {
	  arma::vec sub_left_vector(split_dimensions.n_elem);
	  MakeBoundVector(left_vector, split_dimensions, sub_left_vector);
	  left_bound |= sub_left_vector;
	}
        left++;
      }

      while (matrix(dim, right) >= splitvalue && likely(left <= right)) {
        arma::vec right_vector = matrix.col(right);
	if (split_dimensions.n_elem == matrix.n_rows) {
	  right_bound |= right_vector;
	} else {
	  arma::vec sub_right_vector(split_dimensions.n_elem);
	  MakeBoundVector(right_vector, split_dimensions, sub_right_vector);
	  right_bound |= sub_right_vector;
	}        
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      // swap left and right vector
      matrix.swap_cols(left, right);

      arma::vec left_vector = matrix.col(left);
      arma::vec right_vector = matrix.col(right);

      if (split_dimensions.n_elem == matrix.n_rows) {
	left_bound |= left_vector;
      } else {
	arma::vec sub_left_vector(split_dimensions.n_elem);
	MakeBoundVector(left_vector, split_dimensions, sub_left_vector);
	left_bound |= sub_left_vector;
      }  

      if (split_dimensions.n_elem == matrix.n_rows){
	  right_bound |= right_vector;
      } else {
	arma::vec sub_right_vector(split_dimensions.n_elem);
	MakeBoundVector(right_vector, split_dimensions, sub_right_vector);
	right_bound |= sub_right_vector;
      }  
      
      DEBUG_ASSERT(left <= right);
      right--;
    }

    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename TBound, typename T>
  index_t SelectMatrixPartition(arma::Mat<T>& matrix, 
                                const arma::uvec& split_dimensions,
                                index_t dim,
                                double splitvalue,
                                index_t first,
                                index_t count,
                                TBound& left_bound,
                                TBound& right_bound,
                                arma::Col<index_t>& old_from_new) {
    
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix(dim, left) < splitvalue && likely(left <= right)) {
	arma::vec left_vector = matrix.col(left);
        if (split_dimensions.n_elem == matrix.n_rows) {
	  left_bound |= left_vector;
	} else {
	  arma::vec sub_left_vector(split_dimensions.n_elem);
	  MakeBoundVector(left_vector, split_dimensions, sub_left_vector);
	  left_bound |= sub_left_vector;
	}
        left++;
      }

      while (matrix(dim, right) >= splitvalue && likely(left <= right)) {
        arma::vec right_vector = matrix.col(right);
	if (split_dimensions.n_elem == matrix.n_rows) {
	  right_bound |= right_vector;
	} else {
	  arma::vec sub_right_vector(split_dimensions.n_elem);
	  MakeBoundVector(right_vector, split_dimensions, sub_right_vector);
	  right_bound |= sub_right_vector;
	}        
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      // swap left and right vector
      matrix.swap_cols(left, right);

      arma::vec left_vector = matrix.col(left);
      arma::vec right_vector = matrix.col(right);

      if (split_dimensions.n_elem == matrix.n_rows) {
	left_bound |= left_vector;
      } else {
	arma::vec sub_left_vector(split_dimensions.n_elem);
	MakeBoundVector(left_vector, split_dimensions, sub_left_vector);
	left_bound |= sub_left_vector;
      }  

      if (split_dimensions.n_elem == matrix.n_rows){
	  right_bound |= right_vector;
      } else {
	arma::vec sub_right_vector(split_dimensions.n_elem);
	MakeBoundVector(right_vector, split_dimensions, sub_right_vector);
	right_bound |= sub_right_vector;
      }  
      
      // update indices
      index_t t = old_from_new[left];
      old_from_new[left] = old_from_new[right];
      old_from_new[right] = t;

      DEBUG_ASSERT(left <= right);
      right--;
    }

    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename TKdTree, typename T>
  void SplitKdTreeMidpoint(arma::Mat<T>& matrix,
                           TKdTree *node,
                           index_t leaf_size,
                           arma::Col<T>& old_from_new) {

    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    SelectSplitKdTreeMidpoint(matrix, split_dimensions, node, 
			      leaf_size, old_from_new);
  }
  
  template<typename TKdTree, typename T>
  void SplitKdTreeMidpoint(arma::Mat<T>& matrix,
                           TKdTree *node,
                           index_t leaf_size) {

    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    SelectSplitKdTreeMidpoint(matrix, split_dimensions, node, 
			      leaf_size);
  }

  template<typename TKdTree, typename T>
  void SelectSplitKdTreeMidpoint(arma::Mat<T>& matrix,
                                 const arma::uvec& split_dimensions,
                                 TKdTree *node,
                                 index_t leaf_size, 
                                 arma::Col<index_t>& old_from_new) {
  
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    SelectFindBoundFromMatrix(matrix, split_dimensions, node->begin(), 
			      node->count(), &node->bound());

    if(node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      double max_width = -1;

      for (index_t d = 0; d < split_dimensions.n_elem; d++) {
        double w = node->bound()[d].width();
	
        if (w > max_width) {
          max_width = w;
          split_dim = d;
        }
      }

      double split_val = node->bound()[split_dim].mid();

      if (max_width == 0) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } else {
        left = new TKdTree();
        left->bound().SetSize(split_dimensions.n_elem);

        right = new TKdTree();
        right->bound().SetSize(split_dimensions.n_elem);

        index_t split_col = SelectMatrixPartition(matrix, split_dimensions, 
	    (int) split_dimensions[split_dim], split_val,
            node->begin(), node->count(),
            left->bound(), right->bound(),
            old_from_new);
        
        left->Init(node->begin(), split_col - node->begin());
        right->Init(split_col, node->begin() + node->count() - split_col);

        // This should never happen if max_width > 0
        DEBUG_ASSERT(left->count() != 0 && right->count() != 0);

        SelectSplitKdTreeMidpoint(matrix, split_dimensions, left, leaf_size, 
				  old_from_new);
        SelectSplitKdTreeMidpoint(matrix, split_dimensions, right, leaf_size, 
				  old_from_new);
      }
    }

    node->set_children(matrix, left, right);
  }
  
  template<typename TKdTree, typename T>
  void SelectSplitKdTreeMidpoint(arma::Mat<T>& matrix,
                                 const arma::uvec& split_dimensions,
                                 TKdTree *node,
                                 index_t leaf_size) {
  
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    SelectFindBoundFromMatrix(matrix, split_dimensions, node->begin(), 
			      node->count(), &node->bound());

    if(node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      double max_width = -1;

      for (index_t d = 0; d < split_dimensions.n_elem; d++) {
        double w = node->bound()[d].width();
	
        if (w > max_width) {
          max_width = w;
          split_dim = d;
        }
      }

      double split_val = node->bound()[split_dim].mid();

      if (max_width == 0) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } else {
        left = new TKdTree();
        left->bound().SetSize(split_dimensions.n_elem);

        right = new TKdTree();
        right->bound().SetSize(split_dimensions.n_elem);

        index_t split_col = SelectMatrixPartition(matrix, split_dimensions, 
	    (int) split_dimensions[split_dim], split_val,
            node->begin(), node->count(),
            left->bound(), right->bound());
        
        left->Init(node->begin(), split_col - node->begin());
        right->Init(split_col, node->begin() + node->count() - split_col);

        // This should never happen if max_width > 0
        DEBUG_ASSERT(left->count() != 0 && right->count() != 0);

        SelectSplitKdTreeMidpoint(matrix, split_dimensions, left, leaf_size);
        SelectSplitKdTreeMidpoint(matrix, split_dimensions, right, leaf_size);
      }
    }

    node->set_children(matrix, left, right);
  }

}; // namespace tree_kdtree_private

namespace tree {

/**
 * Creates a KD tree from data, splitting on the midpoint.
 *
 * @experimental
 *
 * This requires you to pass in two vectors which will contain index mappings
 * so you can account for the re-ordering of the matrix.
 *
 * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
 * @param split_dimensions ordering of the dimensions that we should split on;
 *        the first element in this vector will be the first element we split
 *        on, and so on
 * @param leaf_size the maximum points in a leaf
 * @param old_from_new vector that will map new indices to original
 * @param new_from_old vector that will map original indexes to new indices
 */
template<typename TKdTree, typename T>
TKdTree *MakeKdTreeMidpointSelective(arma::Mat<T>& matrix,
                                     const arma::uvec& split_dimensions,
                                     index_t leaf_size,
                                     arma::Col<index_t>& old_from_new,
                                     arma::Col<index_t>& new_from_old) {
  TKdTree *node = new TKdTree();
   
  old_from_new.set_size(matrix.n_cols);
  for (index_t i = 0; i < matrix.n_cols; i++)
    old_from_new[i] = i;
   
  node->Init(0, matrix.n_cols);
  node->bound().SetSize(split_dimensions.n_elem);
  tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
      0, matrix.n_cols, &node->bound());
   
  tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions,
      node, leaf_size, old_from_new);
   
  if(new_from_old) {
    new_from_old.set_size(matrix.n_cols);
    for (index_t i = 0; i < matrix.n_cols; i++) {
      new_from_old[old_from_new[i]] = i;
    }
  }                                                          
  return node;                                               
}

/**
 * Creates a KD tree from data, splitting on the midpoint.
 *
 * @experimental
 *
 * This requires you to pass in one vector that will map new indices to the
 * original.  It does not bother keeping track of a map of the original
 * indices to the new indices.
 *
 * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
 * @param split_dimensions ordering of the dimensions that we should split on;
 *        the first element in this vector will be the first element we split
 *        on, and so on
 * @param leaf_size the maximum points in a leaf
 * @param old_from_new vector that will map new indices to original
 */
template<typename TKdTree, typename T>
TKdTree *MakeKdTreeMidpointSelective(arma::Mat<T>& matrix,
                                     const arma::uvec& split_dimensions,
                                     index_t leaf_size,
                                     arma::Col<index_t>& old_from_new) {
  TKdTree *node = new TKdTree();

  old_from_new.set_size(matrix.n_cols);
  for (index_t i = 0; i < matrix.n_cols; i++)
    old_from_new[i] = i;

  node->Init(0, matrix.n_cols);
  node->bound().SetSize(split_dimensions.n_elem);
  tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
      0, matrix.n_cols, &node->bound());

  tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions,
      node, leaf_size, old_from_new);

  return node;
}

/**
 * Creates a KD tree from data, splitting on the midpoint.
 *
 * @experimental
 *
 * This does not keep track of a map of the old indices to the new indices.
 *
 * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
 * @param split_dimensions ordering of the dimensions that we should split on;
 *        the first element in this vector will be the first element we split
 *        on, and so on
 * @param leaf_size the maximum points in a leaf
 */
template<typename TKdTree, typename T>
TKdTree *MakeKdTreeMidpointSelective(arma::Mat<T>& matrix,
                                     const arma::uvec& split_dimensions,
                                     index_t leaf_size) {
  TKdTree *node = new TKdTree();
   
  node->Init(0, matrix.n_cols);
  node->bound().SetSize(split_dimensions.n_elem);
  tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
      0, matrix.n_cols, &node->bound());
   
  tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions,
      node, leaf_size);
   
  return node;
}

template<typename TKdTree, typename T>
TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix,
                            index_t leaf_size,
                            arma::Col<index_t>& old_from_new,
                            arma::Col<index_t>& new_from_old) {
  // create vector of dimensions that we will split on
  // by default we'll just split the first dimension first, and so on
  arma::uvec split_dimensions(matrix.n_rows);
  for(int i = 0; i < matrix.n_rows; i++)
    split_dimensions[i] = i;

  TKdTree *result;
  result = MakeKdTreeMidpointSelective<TKdTree>(matrix,
      split_dimensions, leaf_size, old_from_new, new_from_old);

  return result;
}

template<typename TKdTree, typename T>
TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix,
                            index_t leaf_size,
                            arma::Col<index_t>& old_from_new) {
  // create vector of dimensions that we will split on
  // by default we'll just split the first dimension first, and so on
  arma::uvec split_dimensions(matrix.n_rows);
  for(index_t i = 0; i < matrix.n_rows; i++)
    split_dimensions[i] = i;

  TKdTree *result;
  result = MakeKdTreeMidpointSelective<TKdTree>(matrix,
      split_dimensions, leaf_size, old_from_new);

  return result;
}

template<typename TKdTree, typename T>
TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, index_t leaf_size) {
  // create vector of dimensions that we will split on
  // by default we'll just split the first dimension first, and so on
  arma::uvec split_dimensions(matrix.n_rows);
  for(int i = 0; i < matrix.n_rows; i++)
    split_dimensions[i] = i;

  TKdTree *result;
  result = MakeKdTreeMidpointSelective<TKdTree>(matrix,
      split_dimensions, leaf_size);

  return result;
}

}; // namespace tree
}; // namespace mlpack

#endif
