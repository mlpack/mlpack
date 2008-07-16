/* Implementation for the regular pointer-style kd-tree builder. */

namespace tree_kdtree_mmap_private {
  Vector MakeBoundVector(Vector point, Vector bound_dimensions){
    int i;
    Vector bound_vector;
    bound_vector.Init(bound_dimensions.length());
    for (i = 0; i < bound_dimensions.length(); i++){
      bound_vector[i] = point[(int)bound_dimensions[i]];
    }
    return bound_vector;
  }


  template<typename TBound>
    void SelectFindBoundFromMatrix(const Matrix& matrix,
      Vector split_dimensions, index_t first, index_t count, TBound *bounds){
    index_t end = first + count;
    for (index_t i = first; i < end; i++) {
      Vector col;
      matrix.MakeColumnVector(i, &col);
      if (split_dimensions.length() == matrix.n_rows()){
	*bounds |= col;
      } else {
	Vector foo = MakeBoundVector(col, split_dimensions);	
	*bounds |= foo;
      }          
    }
  }

  template<typename TBound>
  void FindBoundFromMatrix(const Matrix& matrix,
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

  template<typename TBound>
  index_t SelectMatrixPartition(
      Matrix& matrix, Vector split_dimensions, index_t dim, double splitvalue,
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
	if (split_dimensions.length() == matrix.n_rows()){
	  *left_bound |= left_vector;
	} else {
	  *left_bound |= MakeBoundVector(left_vector, split_dimensions);
	}
        left++;
      }

      while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
        Vector right_vector;
        matrix.MakeColumnVector(right, &right_vector);
	if (split_dimensions.length() == matrix.n_rows()){
	  *right_bound |= right_vector;
	} else {
	  *right_bound |= MakeBoundVector(right_vector, split_dimensions);
	}        
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

      if (split_dimensions.length() == matrix.n_rows()){
	  *left_bound |= left_vector;
      } else {
	  *left_bound |= MakeBoundVector(left_vector, split_dimensions);
      }  

      if (split_dimensions.length() == matrix.n_rows()){
	  *right_bound |= right_vector;
      } else {
	  *right_bound |= MakeBoundVector(right_vector, split_dimensions);
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

  template<typename TKdTree>
  void SplitKdTreeMidpoint(Matrix& matrix,
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

  template<typename TKdTree>
  void SelectSplitKdTreeMidpoint(Matrix& matrix, Vector& split_dimensions,
      TKdTree *node, index_t leaf_size, index_t *old_from_new) {
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
         ptrdiff_t usage1=mmapmm::MemoryManager<false>::allocator_->get_usage();

        typename TKdTree::Bound::StaticBound left_static_bound;
        typename TKdTree::Bound::StaticBound right_static_bound;
        left_static_bound.Init(split_dimensions.length());
        right_static_bound.Init(split_dimensions.length());

        index_t split_col = SelectMatrixPartition(matrix, split_dimensions, 
	     (int)split_dimensions[split_dim], split_val,
            node->begin(), node->count(),
            &left_static_bound, &right_static_bound,
            old_from_new);
        
        VERBOSE_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
            node->begin(), split_col,
            node->begin() + node->count(), (int)split_dimensions[split_dim], 
		    split_val,
            node->bound().get(split_dim).lo,
            node->bound().get(split_dim).hi);

        // This should never happen if max_width > 0
        //DEBUG_ASSERT(left->count() != 0 && right->count() != 0);

        left = new TKdTree();
        left->Init(node->begin(), split_col - node->begin());
        left->bound().Init(split_dimensions.length());
        left->bound().Copy(left_static_bound);
        SelectSplitKdTreeMidpoint(matrix, split_dimensions, left, leaf_size, 
				  old_from_new);
        ptrdiff_t usage2 = mmapmm::MemoryManager<false>::allocator_->get_usage();
        node->stat().set_left_usage(usage2-usage1);
        
        right = new TKdTree();
        right->Init(split_col, node->begin() + node->count() - split_col);
        right->bound().Init(split_dimensions.length());
        right->bound().Copy(right_static_bound);
        SelectSplitKdTreeMidpoint(matrix, split_dimensions, right, leaf_size, 
				  old_from_new);
        ptrdiff_t usage3=mmapmm::MemoryManager<false>::allocator_->get_usage();
        node->stat().set_right_usage(usage3-usage2);
      }
    }

    node->set_children(matrix, left, right);
  }
};
