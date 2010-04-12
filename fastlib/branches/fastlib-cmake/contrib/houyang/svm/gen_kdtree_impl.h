/* Implementation for the regular pointer-style spill kd-tree builder. */

#include "fastlib/fastlib_int.h"

namespace tree_gen_kdtree_private {

  template<typename T, typename TBound>
  void FindBoundFromMatrix(const GenMatrix<T>& matrix,
			   index_t first, index_t count, TBound *bounds) {
    
    index_t end = first + count;
    for (index_t i = first; i < end; i++) {
      GenVector<T> col;
      
      matrix.MakeColumnVector(i, &col);
      *bounds |= col;
    }
  }
  
  template<typename T, typename TBound>
  index_t MatrixPartition(GenMatrix<T>& matrix, index_t dim, double splitvalue,
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
  void SplitGenKdTree(GenMatrix<T>& matrix, TKdTree *node, 
		      index_t leaf_size, index_t *old_from_new,
		      ArrayList< GenVector<index_t> > &o_f_n_maps,
		      index_t &node_id) {

    o_f_n_maps.PushBack();
    o_f_n_maps[node_id].Init(matrix.n_cols());
    o_f_n_maps[node_id].CopyValues(old_from_new);

    /*
    printf("%d\n", node_id);
    for (index_t i=0; i<50; i++)
      printf("%d_ ", (o_f_n_maps[node_id])[i]);
    printf("\n");
    printf("begin=%d count=%d\n", node->begin(), node->count());
    */

    node_id ++;

    TKdTree *left = NULL;
    TKdTree *right = NULL;

    if (node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      T max_width = -1;
      
      for (index_t d = 0; d < matrix.n_rows(); d++) {
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

        index_t split_col = MatrixPartition(matrix, split_dim, split_val,
					    node->begin(), node->count(),
					    &left->bound(), &right->bound(),
					    old_from_new);
	
	VERBOSE_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
		    node->begin(), split_col,
		    node->begin() + node->count(), split_dim, split_val,
		    node->bound().get(split_dim).lo,
		    node->bound().get(split_dim).hi);

	// store the index of the splitting point for this node
	node->set_split_point_idx_old(old_from_new[split_col]);
	
	left->Init(node->begin(), split_col - node->begin(), node_id);
	right->Init(split_col+ 1, node->begin() + node->count() - split_col- 1, node_id);
	// excluding the splitting point from left and right children
	//right->Init(split_col, node->begin() + node->count() - split_col);
	
	
	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, left, leaf_size, old_from_new, o_f_n_maps, node_id);
	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, right, leaf_size, old_from_new, o_f_n_maps, node_id);
      }
    }

    node->set_children(matrix, left, right);
  }

  int SplitDimCompare(const void *col_a, const void *col_b) {
    const double *col_a_ptr = (double*)col_a;
    const double *col_b_ptr = (double*)col_b;
    if (*col_a_ptr> *col_b_ptr)
      return 1;
    else if (*col_a_ptr< *col_b_ptr)
      return -1;
    else
      return 0;
  }

  template<typename T, typename TBound>
  index_t BalancedMatrixPartition(GenMatrix<T>& matrix, index_t dim, double splitvalue,
			  index_t first, index_t count, TBound* left_bound, 
			  TBound* right_bound, ArrayList<index_t> &old_from_new) {
    /*
    // qsort samples according to values in split_dim and take the median sample as splitting one
    Matrix m_sort;
    //index_t n_cols = matrix.n_cols();
    m_sort.Init(2, count);
    for (index_t i=0; i<count; i++) {
      m_sort.set(0, i, matrix.get(dim, first+i));
      m_sort.set(1, i, old_from_new[first+i]);
    }
    qsort(m_sort.prt(), count, 2*sizeof(double), SplitDimCompare);

    // read back the partial-shuffled old_from_new
    for (index_t i=0; i<count; i++) {
      old_from_new[fist+i] = m_sort.get(1, i);
    }

    // partially shuffle the data matrix
    Matrix m_temp;
    m_temp.Copy(matrix.GetColumnPtr(first), matrix.n_rows(), count);
    matrix.set(TODO);

    return index_t(first + count / 2);
    */

    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (likely(left <= right) && matrix.get(dim, left) < splitvalue) {
        GenVector<T> left_vector;
        matrix.MakeColumnVector(left, &left_vector);
        *left_bound |= left_vector;
        left++;
      }

      while (likely(left <= right) && matrix.get(dim, right) >= splitvalue) {
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
      
      //if (old_from_new) {
        index_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
	//}
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    /*
    for (index_t i=first ; i<index_t(first+ count / 2); i++) {
      GenVector<T> left_vector;

      matrix.MakeColumnVector(left, &left_vector);

      *left_bound |= left_vector;
    }

    for (index_t  i=index_t(first+ count / 2); i<first+ count; i++) {
      GenVector<T> right_vector;
      
      matrix.MakeColumnVector(right, &right_vector);

      *right_bound |= right_vector;
    }
    */    

    DEBUG_ASSERT(left == right + 1);

    //return left;
    return index_t(first + count / 2);
    
  }


  template<typename T, typename TKdTree, typename TKdTreeSplitter>
  void SplitBalancedGenKdTree(GenMatrix<T>& matrix, TKdTree *node, 
		      index_t leaf_size, ArrayList<index_t> &old_from_new,
		      ArrayList< GenVector<index_t> > &o_f_n_maps,
		      index_t &node_id) {

    //printf("node_id=%d, count=%d\n", node_id, node->count());
    /*
    printf("%d\n", node_id);
    for (index_t i=0; i<50; i++)
      printf("%d_ ", (o_f_n_maps[node_id])[i]);
    printf("\n");
    printf("begin=%d count=%d\n", node->begin(), node->count());
    */

    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    // Try splitting
    if (node->count() > leaf_size) { // can split this node
      index_t split_dim = BIG_BAD_NUMBER;
      T max_width = -1;
      
      for (index_t d = 0; d < matrix.n_rows(); d++) {
        T w = node->bound().get(d).width();
	//printf("lo=%f, hi=%f, d=%d, w=%f\n",node->bound().get(d).lo, node->bound().get(d).hi, d, w);
	
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

        // Sort data vectors and divide into two parts.
	// This returned split_col is a index in the new shuffled index set.
	index_t split_col_new = BalancedMatrixPartition(matrix, split_dim, split_val,
						    node->begin(), node->count(),
						    &left->bound(), &right->bound(),
						    old_from_new);


	// store the index(old) of the splitting point for the old node
	node->set_split_point_idx_old(old_from_new[split_col_new]);
    



	node_id ++;
	left->Init(node->begin(), split_col_new - node->begin(), node_id);
	
	// left and right children share the same new shuffled index set
	o_f_n_maps.PushBack();
	o_f_n_maps[node_id].Init(matrix.n_cols());
	o_f_n_maps[node_id].CopyValues(old_from_new.begin());
	

	node_id ++;
	right->Init(split_col_new+ 1, node->begin() + node->count() - split_col_new- 1, node_id);

	// left and right children share the same new shuffled index set
	o_f_n_maps.PushBack();
	o_f_n_maps[node_id].Init(matrix.n_cols());
	o_f_n_maps[node_id].CopyValues(old_from_new.begin());


	
	SplitBalancedGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, left, leaf_size, old_from_new, o_f_n_maps, node_id);

	
	SplitBalancedGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, right, leaf_size, old_from_new, o_f_n_maps, node_id);

	}
    }
        
    node->set_children(matrix, left, right);
  }

};
