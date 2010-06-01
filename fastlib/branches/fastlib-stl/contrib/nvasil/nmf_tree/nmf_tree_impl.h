/*
 * =====================================================================================
 * 
 *       Filename:  nmf_tree_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  08/18/2008 01:12:28 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifdef NMF_TREE_IMPL_H_
#define NMF_TREE_IMP_H_

// we use this macro to avoid including this file directly
#ifdef NMF_TREE_H_ 
template<typename TKdTree, typename T>
void NmfTreeConstructor<TKdTree, T>::Init(fx_module *module, 
            ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values) {
    
  module_=fx_submodule(module, "tree");;
  fx_module *l_bfgs_module=fx_submodule(module_, "optimizer");
  fx_module *relaxed_nmf_module=fx_submodule(module_, "l_bfgs");
  rows_.Copy(rows);
  columns_.Copy(columns);
  values_.Copy(values);
  index_t num_of_rows=*std::max_element(rows_.begin(), rows_.end())+1;
  index_t num_of_columns=*std::max_element(columns_.begin(), columns_.end())+1;
  new_dimension_ = fx_param_int(module, "new_dimension", 5);
  data_matrix_.Init(new_dimension_, num_of_rows+num_of_columns);
  w_matrix_.Alias(data_matrix_.ptr(), new_dimension_, num_of_rows_);
  h_matrix_.Alias(data_matrix_.ptr(), new_dimension_, num_of_columns_);  
  
  lower_bound_.Init(new_dimension_, num_of_rows+num_of_columns);
  upper_bound_.Init(new_dimension_, num_of_rows+num_of_columns);
  lower_bound_.SetAll(fx_param_double(module_, "lower_bound", 1e-5));
  upper_bound_.SetAll(fx_param_double(module_, "upper_bound", 1.0));

 
  opt_fun_.Init(relaxed_nmf_module,
                &rows_,
                &columns_,
                &values_,
                &lower_bound, 
                & upper_bound);
  l_bfgs_engine_.Init(&opt_fun, l_bfgs_module_);
  leaf_size_=fx_param_int(module, "leaf_size", 20);
   
  old_from_new_h_.Init(num_of_columns_);
  for (index_t i = 0; i < num_of_rows_; i++) {
    old_from_new_h_[i] = i;
  } 
  old_from_new_w_.Init(num_of_rows_);
  for (index_t i = 0; i < num_of_columns_; i++) {
    old_from_new_w_[i] = i;
  } 
 
}


template<typename TKdTree, typename T>
void NmfTreeConstructor<TKdTree, T>::MakeNmfTree() { 
  parent_= new TKdTree();  
  parent_->Init(0, data_matrix.n_cols());
  MakeNmfTreeMidpointSelective(parent_w_, parent_h_);
}

template<typename TKdTree, typename T>
void NmfTreeConstructor<TKdTree, T>::MakeNmfTreeMidpointSelective(
    TKdTree *node1, TKdTree *node2) {
  GenVector<T> split_dimensions;
  split_dimensions.Init(data_matrix_.n_rows());
  int i;
  for (i = 0; i < data_matrix.n_rows(); i++){
    split_dimensions[i] = i;
  }
  SelectSplitKdTreeMidpoint(node1, node2, split_dimensions);
}

template<typename TKdTree, typename T>
void NmfTreeConstructor<TKdTree, T>::SelectSplitKdTreeMidpoint(TKdTree *node1,
                                 TKdTree *node2, 
                                 const GenVector<T>& split_dimensions) {
  
  TKdTree *left = NULL;
  TKdTree *right = NULL;  
  optimizer.Reset();
  GenMatrix<T> init_matrix;
  opt_fun.GiveInitMatrix(&init_matrix);  
  optimizer_.set(init_matrix);
  optimizer_ComputeLocalOptimumBFGS();
    
  tree_kdtree_private::SelectFindBoundFromMatrix(data_matrix_, 
        split_dimensions, 
        node->begin(), 
			  node->count(), 
        &node->bound());

  if (node1->count() > leaf_size_ && node2->count() > leaf_size_) {
    SplitAndBound(node1, w_matrix_, split_dimensions);
    SplitAndBound(node2, h_matrix_, split_dimensions);
    SelectSplitKdTreeMidpoint(node1->left(), node2->left(), split_dimensions);
    SelectSplitKdTreeMidpoint(node1->right(), node2->right(), split_dimensions);
  } else {
    if (node1->count() > leaf_size_ && node2->count() < leaf_size_) {
      SplitAndBound(node1, w_matrix_, split_dimensions);
      SelectSplitKdTreeMidpoint(node1->left(), node2->left(), split_dimensions);
      SelectSplitKdTreeMidpoint(node1->right(), node2->right(), split_dimensions);
    } else {
      if (node1->count() < leaf_size_ && node2->count() > leaf_size_) {
        SplitAndBound(node2, h_matrix_, split_dimensions);
        SelectSplitKdTreeMidpoint(node1->left(), node2->left(), split_dimensions);
        SelectSplitKdTreeMidpoint(node1->right(), node2->right(), split_dimensions);
      }   
    }
  }
}

template<typename TBound, typename T>
void NmfTreeConstructor<TKdTree, T>::SplitAndBound(TKdTree *node,
    GenMatrix<T> *data_matrix,
    GenVector<T> &split_dimensions) {
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
    node->left->bound().Init(split_dimensions.length());
    node->right = new TKdTree();
    node->right->bound().Init(split_dimensions.length());
    index_t split_col = tree_kdtree_private::SelectMatrixPartition(data_matrix, 
        split_dimensions, 
       (int)split_dimensions[split_dim], split_val,
        node->begin(), node->count(),
        &node->left->bound(), &node->right->bound(),
        old_from_new_);
   
    VERBOSE_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
        node->begin(), split_col,
        node->begin() + node->count(), (int)split_dimensions[split_dim],
		    split_val,
        node->bound().get(split_dim).lo,
        node->bound().get(split_dim).hi);

    node->left->Init(node->begin(), split_col - node->begin());
    node->right->Init(split_col, node->begin() + node->count() - split_col);

   // This should never happen if max_width > 0
   DEBUG_ASSERT(node->left->count() != 0 && node->right->count() != 0);
}

template<typename TBound, typename T>
index_t NmfTreeConstructor<TKdTree, T>::SelectMatrixPartition(GenMatrix<T>& matrix, 
  const Vector& split_dimensions, index_t dim, double splitvalue,
  index_t first, index_t count, TBound* left_bound, TBound* right_bound) {
    
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
	    if (split_dimensions.length() == matrix.n_rows()) {
	      *left_bound |= left_vector;
	    } else {
	      GenVector<T> sub_left_vector;
	      sub_left_vector.Init(split_dimensions.length());
	      MakeBoundVector(left_vector, split_dimensions, &sub_left_vector);
	      *left_bound |= sub_left_vector;
	    }
      left++;
    }

    while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
      GenVector<T> right_vector;
      matrix.MakeColumnVector(right, &right_vector);
	    if (split_dimensions.length() == matrix.n_rows()) {
	      *right_bound |= right_vector;
	    } else {
	      GenVector<T> sub_right_vector;
	      sub_right_vector.Init(split_dimensions.length());
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
    if (split_dimensions.length() == matrix.n_rows()) {
	    *left_bound |= left_vector;
    } else {
	    GenVector<T> sub_left_vector;
	    sub_left_vector.Init(split_dimensions.length());
	    MakeBoundVector(left_vector, split_dimensions, &sub_left_vector);
	    *left_bound |= sub_left_vector;
    }  

    if (split_dimensions.length() == matrix.n_rows()){
	    *right_bound |= right_vector;
    } else {
	    GenVector<T> sub_right_vector;
	    sub_right_vector.Init(split_dimensions.length());
	    MakeBoundVector(right_vector, split_dimensions, &sub_right_vector);
	    *right_bound |= sub_right_vector;
    }   
      
    index_t t = old_from_new[left];
    old_from_new[left] = old_from_new[right];
    old_from_new[right] = t;
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

 
#endif // NMF_TREE_H_
#endif // NMF_TREE_IMPL_H_
