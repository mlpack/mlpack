/* Implementation for the regular pointer-style kd-tree builder. */

namespace tree_kdtree_private {
  template<typename T>
    void MakeBoundVector(const GenVector<T>& point, 
      const Vector& bound_dimensions, GenVector<T>* bound_vector){
    int i;    
    for (i = 0; i < bound_dimensions.length(); i++){
      (*bound_vector)[i] = point[(int)bound_dimensions[i]];
    }
  }
  

  template<typename TBound, typename T>
    void SelectFindBoundFromMatrix(const GenMatrix<T>& matrix,
      const Vector& split_dimensions, index_t first, index_t count, 
      TBound *bounds){
    index_t end = first + count;
    for (index_t i = first; i < end; i++) {
      GenVector<T> col;
      matrix.MakeColumnVector(i, &col);
      if (split_dimensions.length() == matrix.n_rows()){
	*bounds |= col;
      } else {
	GenVector<T> sub_col;
	sub_col.Init(split_dimensions.length());
	MakeBoundVector(col, split_dimensions, &sub_col);	
	*bounds |= sub_col;
      }          
    }
  }

  template<typename TBound, typename T>
  void FindBoundFromMatrix(const GenMatrix<T>& matrix,
      index_t first, index_t count, TBound *bounds){
    Vector split_dimensions;
    split_dimensions.Init(matrix.n_rows());
    int i;
    for (i = 0; i < matrix.n_rows(); i++){
      split_dimensions[i] = i;
    }
    SelectFindBoundFromMatrix(matrix, split_dimensions, first, count, bounds);
  }

   template<typename TBound>
   index_t MatrixPartition(
      Matrix& matrix, index_t dim, double splitvalue,
      index_t first, index_t count,
      TBound* left_bound, TBound* right_bound,
      index_t *old_from_new) {
     Vector split_dimensions;
     split_dimensions.Init(matrix.n_rows());
     int i;
     for (i = 0; i < matrix.n_rows(); i++){
       split_dimensions[i] = i;
     }
     index_t split_point = SelectMatrixPartition(matrix, split_dimensions, 
      dim, splitvalue, first, count, left_bound, right_bound, old_from_new);
     return split_point;
   }

  template<typename TBound, typename T>
  index_t SelectMatrixPartition(GenMatrix<T>& matrix, 
    const Vector& split_dimensions, index_t dim, double splitvalue,
    index_t first, index_t count, TBound* left_bound, TBound* right_bound,
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
        GenVector<T> left_vector;
        matrix.MakeColumnVector(left, &left_vector);
	if (split_dimensions.length() == matrix.n_rows()){
	  *left_bound |= left_vector;
	} else {
	  GenVector<T> sub_left_vector;
	  MakeBoundVector(left_vector, split_dimensions, &sub_left_vector);
	  *left_bound |= sub_left_vector;
	}
        left++;
      }

      while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
        GenVector<T> right_vector;
        matrix.MakeColumnVector(right, &right_vector);
	if (split_dimensions.length() == matrix.n_rows()){
	  *right_bound |= right_vector;
	} else {
	  GenVector<T> sub_right_vector;
	  MakeBoundVector(right_vector, split_dimensions, &sub_right_vector);
	  *right_bound |= sub_right_vector;
	}        
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

      if (split_dimensions.length() == matrix.n_rows()){
	  *left_bound |= left_vector;
      } else {
	GenVector<T> sub_left_vector;
	MakeBoundVector(left_vector, split_dimensions, &sub_left_vector);
	*left_bound |= sub_left_vector;
      }  

      if (split_dimensions.length() == matrix.n_rows()){
	  *right_bound |= right_vector;
      } else {
	GenVector<T> sub_right_vector;
	MakeBoundVector(right_vector, split_dimensions, &sub_right_vector);
	*right_bound |= sub_right_vector;
      }  
     
      
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

  template<typename TKdTree, typename T>
  void SplitKdTreeMidpoint(GenMatrix<T>& matrix,
      TKdTree *node, index_t leaf_size, index_t *old_from_new){

    Vector split_dimensions;
    split_dimensions.Init(matrix.n_rows());
    int i;
    for (i = 0; i < matrix.n_rows(); i++){
      split_dimensions[i] = i;
    }
    SelectSplitKdTreeMidpoint(matrix, split_dimensions, node, 
			      leaf_size, old_from_new);
  }
  

  template<typename TKdTree, typename T>
    void SelectSplitKdTreeMidpoint(GenMatrix<T>& matrix,
      const Vector& split_dimensions, TKdTree *node, index_t leaf_size, 
      index_t *old_from_new) {
  
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    SelectFindBoundFromMatrix(matrix, split_dimensions, node->begin(), 
			      node->count(), &node->bound());

    if (node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      double max_width = -1;

      for (index_t d = 0; d < split_dimensions.length(); d++) {
        double w = node->bound().get(d).width();
	
        if (w > max_width) {	
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
        left->bound().Init(split_dimensions.length());

        right = new TKdTree();
        right->bound().Init(split_dimensions.length());

        index_t split_col = SelectMatrixPartition(matrix, split_dimensions, 
	     (int)split_dimensions[split_dim], split_val,
            node->begin(), node->count(),
            &left->bound(), &right->bound(),
            old_from_new);
        
        VERBOSE_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
            node->begin(), split_col,
            node->begin() + node->count(), (int)split_dimensions[split_dim],
		    split_val,
            node->bound().get(split_dim).lo,
            node->bound().get(split_dim).hi);

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
};
