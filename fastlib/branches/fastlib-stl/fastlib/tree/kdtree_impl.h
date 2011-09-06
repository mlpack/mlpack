/* Implementation for the regular pointer-style kd-tree builder. */

#ifndef TREE_KDTREE_IMPL_H
#define TREE_KDTREE_IMPL_H

#include "../base/arma_compat.h"
#include "../fx/io.h"

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
    index_t end = first + count; // Set index correctly for loop.
    // Loop over each point this bound contains.
    for(index_t i = first; i < end; i++) {
      // If we are using all dimensions to split on, widen the bound in all
      // dimensions.
      if (split_dimensions.n_elem == matrix.n_rows) {
	*bounds |= matrix.col(i);
      } else {
        // Otherwise, only use the dimensions specified in the split_dimensions
        // vector.
        arma::vec sub_col(split_dimensions.n_elem);
	MakeBoundVector(matrix.col(i), split_dimensions, sub_col);
	*bounds |= sub_col;
      }          
    }
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
     *
     * We want all the points on the left to be less than the splitvalue (in the
     * splitting dimension) and on the right to be greater than or equal to
     * splitvalue.
     */
    for (;;) {
      // See how many of the points on the left are correct.  When they are
      // correct, increase the bound of the left node accordingly.  When we
      // encounter one that isn't correct, move to the right side.
      while (matrix(dim, left) < splitvalue && (left <= right)) {
        if (split_dimensions.n_elem == matrix.n_rows) {
	  left_bound |= matrix.unsafe_col(left);
	} else {
          // Ignore certain dimensions when updating the bound.
	  arma::vec sub_left_vector(split_dimensions.n_elem);
	  MakeBoundVector(matrix.unsafe_col(left), split_dimensions,
              sub_left_vector);
	  left_bound |= sub_left_vector;
	}
        left++;
      }

      // See how many of the points on the right are correct.  When they are
      // correct, increase the bound of the right node accordingly.  When we
      // encounter one that isn't correct, move to the swapping.
      while (matrix(dim, right) >= splitvalue && (left <= right)) {
	if (split_dimensions.n_elem == matrix.n_rows) {
	  right_bound |= matrix.unsafe_col(right);
	} else {
          // Ignore certain dimensions when updating the bound.
	  arma::vec sub_right_vector(split_dimensions.n_elem);
	  MakeBoundVector(matrix.unsafe_col(right), split_dimensions,
              sub_right_vector);
	  right_bound |= sub_right_vector;
	}        
        right--;
      }

      if (left > right) {
        // The procedure is finished.
        break;
      }

      // Swap left and right vector; both are wrong.
      matrix.swap_cols(left, right);
      
      mlpack::IO::Assert(left <= right);
    }

    mlpack::IO::Assert(left == right + 1);

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
                                std::vector<index_t>& old_from_new) {
    
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     *
     * We want all the points on the left to be less than the splitvalue (in the
     * splitting dimension) and on the right to be greater than or equal to
     * splitvalue.
     */
    for (;;) {
      // See how many of the points on the left are correct.  When they are
      // correct, increase the bound of the left node accordingly.  When we
      // encounter one that isn't correct, move to the right side.
      while (matrix(dim, left) < splitvalue && (left <= right)) {
        if (split_dimensions.n_elem == matrix.n_rows) {
	  left_bound |= matrix.unsafe_col(left);
	} else {
          // Ignore certain dimensions when updating the bound.
	  arma::vec sub_left_vector(split_dimensions.n_elem);
	  MakeBoundVector(matrix.unsafe_col(left), split_dimensions,
              sub_left_vector);
	  left_bound |= sub_left_vector;
	}
        left++;
      }

      // See how many of the points on the right are correct.  When they are
      // correct, increase the bound of the right node accordingly.  When we
      // encounter one that isn't correct, move to the swapping.
      while (matrix(dim, right) >= splitvalue && (left <= right)) {
	if (split_dimensions.n_elem == matrix.n_rows) {
	  right_bound |= matrix.unsafe_col(right);
	} else {
          // Ignore certain dimensions when updating the bound.
	  arma::vec sub_right_vector(split_dimensions.n_elem);
	  MakeBoundVector(matrix.unsafe_col(right), split_dimensions,
              sub_right_vector);
	  right_bound |= sub_right_vector;
	}        
        right--;
      }

      if (left > right) {
        // The procedure is done.
        break;
      }

      // Swap left and right vector.
      matrix.swap_cols(left, right);

      // Update indices for what we changed.
      index_t t = old_from_new[left];
      old_from_new[left] = old_from_new[right];
      old_from_new[right] = t;

      mlpack::IO::Assert(left <= right);
    }

    mlpack::IO::Assert(left == right + 1);

    return left;
  }

  template<typename TKdTree, typename T>
  void SelectSplitKdTreeMidpoint(arma::Mat<T>& matrix,
                                 const arma::uvec& split_dimensions,
                                 TKdTree *node,
                                 index_t leaf_size, 
                                 std::vector<index_t>& old_from_new) {
  
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    SelectFindBoundFromMatrix(matrix, split_dimensions, node->begin(), 
			      node->count(), &node->bound());

    if(node->count() > leaf_size) {
      index_t split_dim = (index_t)~0;
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
        mlpack::IO::Assert(left->count() != 0 && right->count() != 0);

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
    // Set the children we will be assigning to NULL, for now.
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    // Set the bounds of the node we are working with correctly.
    SelectFindBoundFromMatrix(matrix, split_dimensions, node->begin(), 
			      node->count(), &node->bound());

    // If we have enough points to split, let's do that.
    if(node->count() > leaf_size) {
      // Set up the split dimension.  We find the dimension with the largest
      // width to split on.
      index_t split_dim = (index_t)~0;
      double max_width = -1;

      for (index_t d = 0; d < split_dimensions.n_elem; d++) {
        double w = node->bound()[d].width();
	
        if (w > max_width) {
          max_width = w;
          split_dim = d;
        }
      }

      // And we split in the middle of that dimension.
      double split_val = node->bound()[split_dim].mid();

      if (max_width == 0) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } else {
        // Initialize the left node and set the dimension of its bound.
        left = new TKdTree();
        left->bound().SetSize(split_dimensions.n_elem);

        // Initialize the right node and set the dimension of its bound.
        right = new TKdTree();
        right->bound().SetSize(split_dimensions.n_elem);

        // Find the column we will be splitting on.
        index_t split_col = SelectMatrixPartition(matrix, split_dimensions, 
	    (int) split_dimensions[split_dim], split_val,
            node->begin(), node->count(),
            left->bound(), right->bound());
        
        // Set the sizes of the left and right children nodes correctly.
        left->Init(node->begin(), split_col - node->begin());
        right->Init(split_col, node->begin() + node->count() - split_col);

        // This should never happen if max_width > 0
        mlpack::IO::Assert(left->count() != 0 && right->count() != 0);

        // Recurse into setting those child nodes correctly.
        SelectSplitKdTreeMidpoint(matrix, split_dimensions, left, leaf_size);
        SelectSplitKdTreeMidpoint(matrix, split_dimensions, right, leaf_size);
      }
    }

    // Set the children of the main node (they may be NULL if we didn't or
    // couldn't split).
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
                                     std::vector<index_t>& old_from_new,
                                     std::vector<index_t>& new_from_old) {
  TKdTree *node = new TKdTree();
   
  old_from_new.resize(matrix.n_cols);
  for (index_t i = 0; i < matrix.n_cols; i++)
    old_from_new[i] = i;
   
  node->Init(0, matrix.n_cols);
  node->bound().SetSize(split_dimensions.n_elem);
  tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
      0, matrix.n_cols, &node->bound());
   
  tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions,
      node, leaf_size, old_from_new);
   
  if(new_from_old) {
    new_from_old.resize(matrix.n_cols);
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
                                     std::vector<index_t>& old_from_new) {
  TKdTree *node = new TKdTree();

  old_from_new.resize(matrix.n_cols);
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
                            std::vector<index_t>& old_from_new,
                            std::vector<index_t>& new_from_old) {
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
                            std::vector<index_t>& old_from_new) {
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
