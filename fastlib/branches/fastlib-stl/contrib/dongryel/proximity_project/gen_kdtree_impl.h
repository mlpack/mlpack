/* Implementation for the regular pointer-style spill kd-tree builder. */

#include "fastlib/fastlib_int.h"

namespace tree_gen_kdtree_private {

  template<typename T, typename TBound>
  void FindBoundFromMatrix(const GenMatrix<T>& matrix,
			   size_t first, size_t count, TBound *bounds) {
    
    size_t end = first + count;
    for (size_t i = first; i < end; i++) {
      GenVector<T> col;
      
      matrix.MakeColumnVector(i, &col);
      *bounds |= col;
    }
  }
  
  template<typename T, typename TBound>
  size_t MatrixPartition(GenMatrix<T>& matrix, size_t dim, double splitvalue,
			  size_t first, size_t count, TBound* left_bound, 
			  TBound* right_bound, size_t *old_from_new) {
    
    size_t left = first;
    size_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix.get(dim, left) < splitvalue && likely(left <= right)) {
        GenVector<T> left_vector;
        matrix.MakeColumnVector(left, &left_vector);
        *left_bound |= left_vector;
        left++;
      }

      while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
        GenVector<T> right_vector;
        matrix.MakeColumnVector(right, &right_vector);
        *right_bound |= right_vector;
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      GenVector<T> left_vector;
      GenVector<T> right_vector;

      matrix.MakeColumnVector(left, &left_vector);
      matrix.MakeColumnVector(right, &right_vector);

      left_vector.SwapValues(&right_vector);
      
      *left_bound |= left_vector;
      *right_bound |= right_vector;
      
      if (old_from_new) {
        size_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    
    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename T, typename TKdTree, typename TKdTreeSplitter>
  void SplitGenKdTree(GenMatrix<T>& matrix, TKdTree *node, 
		      size_t leaf_size, size_t *old_from_new) {

    TKdTree *left = NULL;
    TKdTree *right = NULL;

    if (node->count() > leaf_size) {
      size_t split_dim = BIG_BAD_NUMBER;
      T max_width = -1;
      
      for (size_t d = 0; d < matrix.n_rows(); d++) {
        T w = node->bound().get(d).width();
	
        if (unlikely(w > max_width)) {
          max_width = w;
          split_dim = d;
        }
      }
      
      // choose the split value along the dimension to be splitted
      double split_val = 
	TKdTreeSplitter::ChooseKdTreeSplitValue(matrix, node, split_dim);

      if (max_width < DBL_EPSILON) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } 
      else {
        left = new TKdTree();
        left->bound().Init(matrix.n_rows());
	
        right = new TKdTree();
        right->bound().Init(matrix.n_rows());

        size_t split_col = MatrixPartition(matrix, split_dim, split_val,
					    node->begin(), node->count(),
					    &left->bound(), &right->bound(),
					    old_from_new);
	
	VERBOSE_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
		    node->begin(), split_col,
		    node->begin() + node->count(), split_dim, split_val,
		    node->bound().get(split_dim).lo,
		    node->bound().get(split_dim).hi);

	left->Init(node->begin(), split_col - node->begin());
	right->Init(split_col, node->begin() + node->count() - split_col);
	
	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, left, leaf_size, old_from_new);
	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, right, leaf_size, old_from_new);
      }
    }

    node->set_children(matrix, left, right);
  }
};
