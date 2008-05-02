/* Implementation for the regular pointer-style kd-tree builder. */

#include "fastlib/fastlib_int.h"

namespace tree_gen_kdtree_private {

  template<typename T, typename TBound>
  void FindBoundFromMatrix(const GenMatrix<T>& lower_limit_matrix,
			   const GenMatrix<T>& upper_limit_matrix,
			   index_t first, index_t count, TBound *bounds) {
    
    index_t end = first + count;
    for (index_t i = first; i < end; i++) {
      GenVector<T> col;
      
      lower_limit_matrix.MakeColumnVector(i, &col);
      *bounds |= col;
      
      col.Destruct();
      upper_limit_matrix.MakeColumnVector(i, &col);
      *bounds |= col;
    }
  }
  
  template<typename T, typename TBound>
  index_t MatrixPartition(GenMatrix<T>& lower_limit_matrix, 
			  GenMatrix<T>& upper_limit_matrix,
			  index_t split_matrix,
			  index_t dim, double splitvalue,
			  index_t first, index_t count, TBound* left_bound, 
			  TBound* right_bound, index_t *old_from_new) {

    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {

      // If the lower limit is at most the split value, then put it in
      // the left.
      double left_split = (split_matrix == 0) ?
	lower_limit_matrix.get(dim, left):upper_limit_matrix.get(dim, left);
      double right_split = (split_matrix == 0) ?
	lower_limit_matrix.get(dim, right):upper_limit_matrix.get(dim, right);
      while (left_split < splitvalue && likely(left <= right)) {
        GenVector<T> left_vector;
        lower_limit_matrix.MakeColumnVector(left, &left_vector);
        *left_bound |= left_vector;
	left_vector.Destruct();
	upper_limit_matrix.MakeColumnVector(left, &left_vector);
	*left_bound |= left_vector;
        left++;
	left_split = (split_matrix == 0) ?
	  lower_limit_matrix.get(dim, left):upper_limit_matrix.get(dim, left);
      }

      // If the upper limit is at least the split value, then put it
      // in the right.
      while (right_split >= splitvalue && likely(left <= right)) {
        GenVector<T> right_vector;
	lower_limit_matrix.MakeColumnVector(right, &right_vector);
	*right_bound |= right_vector;
	right_vector.Destruct();
        upper_limit_matrix.MakeColumnVector(right, &right_vector);
        *right_bound |= right_vector;
        right--;
	right_split = (split_matrix == 0) ?
	  lower_limit_matrix.get(dim, right):
	  upper_limit_matrix.get(dim, right);
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      GenVector<T> left_vector;
      GenVector<T> right_vector;

      // First swap the lower limits.
      lower_limit_matrix.MakeColumnVector(left, &left_vector);
      lower_limit_matrix.MakeColumnVector(right, &right_vector);

      left_vector.SwapValues(&right_vector);
      
      *left_bound |= left_vector;
      *right_bound |= right_vector;
      
      // Then swap the upper limits.
      left_vector.Destruct();
      right_vector.Destruct();
      upper_limit_matrix.MakeColumnVector(left, &left_vector);
      upper_limit_matrix.MakeColumnVector(right, &right_vector);

      left_vector.SwapValues(&right_vector);
      
      *left_bound |= left_vector;
      *right_bound |= right_vector;
      
      // Swap indices...
      if (old_from_new) {
        index_t t = old_from_new[left];
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
  void SplitGenKdTree(GenMatrix<T>& lower_limit_matrix, 
		      GenMatrix<T>& upper_limit_matrix, TKdTree *node,
		      index_t leaf_size, index_t *old_from_new) {

    TKdTree *left = NULL;
    TKdTree *right = NULL;

    if (node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      T max_width = -1;
      double split_val = -1;
      index_t split_matrix = -1;
      
      for (index_t d = 0; d < lower_limit_matrix.n_rows(); d++) {
	T min_coord = 1;
	T max_coord = 0;
	for(index_t p = node->begin(); p < node->end(); p++) {
	  if(p == node->begin()) {
	    min_coord = max_coord = lower_limit_matrix.get(d, p);
	  }
	  else {
	    min_coord = std::min(min_coord, lower_limit_matrix.get(d, p));
	    max_coord = std::max(max_coord, lower_limit_matrix.get(d, p));
	  }
	}
        T w = max_coord - min_coord;
        for(index_t p = node->begin(); p < node->end(); p++) {
	  if(p == node->begin()) {
	    min_coord = max_coord = upper_limit_matrix.get(d, p);
	  }
	  else {
	    min_coord = std::min(min_coord, upper_limit_matrix.get(d, p));
	    max_coord = std::max(max_coord, upper_limit_matrix.get(d, p));
	  }
        }
	T w2 = max_coord - min_coord;
	if(w > w2) {
	  split_matrix = 0;
	}
	else {
	  split_matrix = 1;
	}

	w = std::max(w2, (T) (max_coord - min_coord));
		
        if (unlikely(w > max_width)) {
          max_width = w;
          split_dim = d;
	  split_val = 0.5 * (max_coord + min_coord);
        }
      }

      if (max_width < DBL_EPSILON) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } 
      else {
        left = new TKdTree();
        left->bound().Init(lower_limit_matrix.n_rows());
	
        right = new TKdTree();
        right->bound().Init(lower_limit_matrix.n_rows());

        index_t split_col = 
	  MatrixPartition(lower_limit_matrix, upper_limit_matrix, split_matrix,
			  split_dim, split_val,
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
	  (lower_limit_matrix, upper_limit_matrix, left, leaf_size, 
	   old_from_new);
	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (lower_limit_matrix, upper_limit_matrix, right, leaf_size, 
	   old_from_new);
      }
    }

    node->set_children(lower_limit_matrix, left, right);
  }
};
