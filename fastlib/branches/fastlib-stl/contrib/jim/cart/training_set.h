#ifndef TRAINING_SET_H
#define TRAINING_SET_H

#include "fastlib/fastlib.h"


class TrainingSet{ 

 private:
  Dataset data_;
  Matrix* data_matrix_;
  ArrayList<Vector> order_;  
  ArrayList<Vector> back_order_; 
  ArrayList<int> old_from_new_, new_from_old_;
  int n_features_, n_points_, target_types_;


  int SortOrdinalFeature_(int dim, int start_, int stop_){   
    if (stop_ - start_ == 1){
      if (!isnan(data_matrix_->get(dim, start_))){
	order_[dim][start_] = -1;
	return start_;
      } else {
	order_[dim][start_] = -2;
	return -2;	
      }      
    } 
    int halfway = (start_ + stop_) / 2;      
    int left_start = SortOrdinalFeature_(dim, start_, halfway);
    int right_start = SortOrdinalFeature_(dim, halfway, stop_);     
    // Merge Results
    return MergeDim_(dim, left_start, right_start);
  }

  int MergeDim_(int dim, int left_start, int right_start){
    int merge_start, left, right;
    if (left_start >= 0){
      if (right_start >= 0 ) {
	if (data_matrix_->get(dim, left_start) < 
	    data_matrix_->get(dim, right_start)){
	  merge_start = left_start;
	  left = (int)order_[dim][left_start];
	  right = right_start;
	} else {
	  merge_start = right_start;
	  right = (int)order_[dim][right_start];
	  left = left_start; 
	}
	int current = merge_start;
	while(left >= 0 & right >= 0){
	  if (data_matrix_->get(dim, right) < data_matrix_->get(dim, left)){
	    order_[dim][current] = right;
	    current = right;
	    right = (int)order_[dim][right]; 
	  } else {
	    order_[dim][current] = left;
	    current = left;
	    left = (int)order_[dim][left];
	  }
	}
	if (left >= 0) {
	  order_[dim][current] = left;
	} else {
	  order_[dim][current] = right; 
	}
	return merge_start;
      } else {
	return left_start;
      }
    } else {
      if (right_start >= 0){
	return right_start;
      } else {
	return -1;
      }
    }    
  }



  //////////////////////// Constructors ///////////////////////////////////////

  FORBID_ACCIDENTAL_COPIES(TrainingSet);

 public: 

  TrainingSet(){    
  }

  ~TrainingSet(){
  }

  ////////////////////// Helper Functions /////////////////////////////////////

  void Init(const char* fp, Vector &firsts){   
    data_.InitFromFile(fp);
    // Make linked list representing sorting of ordered vars
    data_matrix_ = &data_.matrix();
    n_features_ = data_.n_features();
    n_points_ = data_.n_points();   
    order_.Init(n_features_);
    back_order_.Init(n_features_);     
    firsts.Init(n_features_);
    int i;
    DatasetInfo meta_data = data_.info();
    const DatasetFeature* current_feature;
    for (i = 0; i < n_features_; i++){
      current_feature = &meta_data.feature(i);      
      if (current_feature->type() != 2 ){
	order_[i].Init(n_points_); 
	back_order_[i].Init(n_points_);
	back_order_[i].SetAll(-2);
	firsts[i] = SortOrdinalFeature_(i, 0, n_points_);
	int j_old = (int)firsts[i], j_cur = (int)order_[i][(int)firsts[i]];
	back_order_[i][j_old] = -1;
	while(j_cur >= 0){
	  back_order_[i][j_cur] = j_old;
	  j_old = j_cur;
	  j_cur = (int)order_[i][j_cur];
	} 	
      } else {
	order_[i].Init(0);
	back_order_[i].Init(0);
      }
    }    
    old_from_new_.Init(n_points_);
    new_from_old_.Init(n_points_);
    for (int i = 0; i < n_points_; i++){
      old_from_new_[i] = i;
      new_from_old_[i] = i;
    }
  }
  
  /*
   * Initialization for features and target in separate files.
   */
  void InitLabels(const char* fp){   
    data_.InitFromFile(fp);
   
    // Make linked list representing sorting of ordered vars
    n_features_ = data_.n_features();
    n_points_ = data_.n_points();  
    old_from_new_.Init(n_points_);
    new_from_old_.Init(n_points_);
    for (int i = 0; i < n_points_; i++){
      old_from_new_[i] = i;
      new_from_old_[i] = i;
    }	 
  }

  void InitLabels2(const char* fl, Vector &firsts, Matrix* data_in){
    Matrix labels_;
    data::Load(fl, &labels_);
    data_matrix_ = data_in;
    target_types_ = 0;
    for (int i = 0; i < n_points_; i++){
      Vector temp;
      data_.matrix().MakeColumnVector(i, &temp);
      for (int j = 0; j < n_features_; j++){
	data_matrix_->set(j,i, temp[j]);
      }      
      data_matrix_->set(n_features_, i, labels_.get(0, i));
      if (labels_.get(0,i) >= target_types_){
	target_types_ = (int)labels_.get(0,i)+1;
      }
    }    
    n_features_++;
    order_.Init(n_features_);
    back_order_.Init(n_features_);     
    firsts.Init(n_features_);
    int i;
    DatasetInfo meta_data = data_.info();
    const DatasetFeature* current_feature;
    for (i = 0; i < n_features_-1; i++){
      current_feature = &meta_data.feature(i);      
      if (current_feature->type() != 2 ){
	order_[i].Init(n_points_); 
	back_order_[i].Init(n_points_);
	back_order_[i].SetAll(-2);
	firsts[i] = SortOrdinalFeature_(i, 0, n_points_);
	int j_old = (int)firsts[i], j_cur = (int)order_[i][(int)firsts[i]];
	back_order_[i][j_old] = -1;
	while(j_cur >= 0){
	  back_order_[i][j_cur] = j_old;
	  j_old = j_cur;
	  j_cur = (int)order_[i][j_cur];
	} 	
      } else {
	order_[i].Init(0);
	back_order_[i].Init(0);
      }
    }       
    order_[n_features_-1].Init(0);
    back_order_[n_features_-1].Init(0);
  }


  void Merge(Vector& left_firsts, Vector& right_firsts, Vector* new_firsts){
    Vector firsts;
    firsts.Init(n_features_);
    firsts.SetZero();
    for (int i = 0; i < order_.size(); i++){
      if (order_[i].length() > 0){
	firsts[i] = MergeDim_(i, (int)left_firsts[i], (int)right_firsts[i]);
	back_order_[i].SetAll(-2);
	int j_old = (int)firsts[i], j_cur = (int)order_[i][(int)firsts[i]];
	back_order_[i][j_old] = -1;
	while(j_cur >= 0){
	  back_order_[i][j_cur] = j_old;
	  j_old = j_cur;
	  j_cur = (int)order_[i][j_cur];
	} 	
      }
    }
    la::ScaleOverwrite(1.0, firsts, new_firsts);
  }


  void Swap(int left, int right, Vector* firsts){
    Vector left_vector;
    Vector right_vector;
    Vector new_firsts;
    la::ScaleInit(1.0, *firsts, &new_firsts);
    
    data_matrix_->MakeColumnVector(left, &left_vector);
    data_matrix_->MakeColumnVector(right, &right_vector);
    
    int temp = old_from_new_[left];
    old_from_new_[left] = old_from_new_[right];
    old_from_new_[right] = temp;
    
    new_from_old_[temp] = right;
    new_from_old_[old_from_new_[left]] = left;
    
    left_vector.SwapValues(&right_vector);
    // Update linked lists
    for (int i = 0; i < n_features_; i++){
      if (likely(order_[i].length() > 0)){	  	 
	if (likely(back_order_[i][right] >= 0)){
	  if (back_order_[i][left] >= 0){
	    order_[i][(int)back_order_[i][right]] = left;
	    order_[i][(int)back_order_[i][left]] = right;	      
	  } else {
	    new_firsts[i] = right;
	    order_[i][(int)back_order_[i][right]] = left;
	  }
	} else {
	  new_firsts[i] = left;
	  order_[i][(int)back_order_[i][left]] = right;
	}
	if (likely(order_[i][right] >= 0)){
	  back_order_[i][(int)order_[i][right]] = left;
	}
	if (likely(order_[i][left] >= 0)){
	  back_order_[i][(int)order_[i][left]] = right;	  
	}
      }	
      double temp = back_order_[i][left];
      back_order_[i][left] = back_order_[i][right];
      back_order_[i][right] = temp;

      temp = order_[i][left];
      order_[i][left] = order_[i][right];
      order_[i][right] = temp;   
    }
    la::ScaleOverwrite(1.0, new_firsts, firsts);
  }


  // This function swaps columns of our data matrix, to represent the
  // partition into left and right nodes.
  index_t MatrixPartition(index_t start, index_t stop, Vector& split, 
	 Vector& firsts, Vector* firsts_l_out, Vector* firsts_r_out){
    Vector firsts_l;
    Vector firsts_r;
    // n_features_ = data_.n_features();
    firsts_l.Init(n_features_);
    firsts_r.Init(n_features_);

    int i;          
    int current_index;
    for (i = 0; i < n_features_; i++) {
      int right_index = -1, left_index = -1;
      if (order_[i].length() > 0) {
	current_index = (int)firsts[i];
	while (current_index >= 0) {
	  if (split[current_index - start] == 1){
	    if (left_index < 0){
	      firsts_l[i] = current_index;	       
	    } else {
	       order_[i][left_index] = current_index;
	    }
	    back_order_[i][current_index] = left_index;
	    left_index = current_index;	    
	  } else {
	    if (right_index < 0) {
	      firsts_r[i] = current_index;	      
	    } else {
	      order_[i][right_index] = current_index;
	    }
	    back_order_[i][current_index] = right_index;	   
	    right_index = current_index;	   
	   }
	  current_index = (int)order_[i][current_index];
	}
	if (left_index >= 0){
	  order_[i][left_index] = -1;
	}
	if (right_index >= 0){
	  order_[i][right_index] = -1;	 
	} 
	
      }
    }
   
    index_t left = start;
    index_t right = stop-1;
    for(;;){
      while (split[left - start] > 0 && likely(left <= right)){
        left++;
      }
      while (likely(left <= right) && split[right - start] < 1){
        right--;
      }
      
      if (unlikely(left > right)){
	break;
      }
   
      Vector left_vector;
      Vector right_vector;
      
      data_matrix_->MakeColumnVector(left, &left_vector);
      data_matrix_->MakeColumnVector(right, &right_vector);

      int temp = old_from_new_[left];
      old_from_new_[left] = old_from_new_[right];
      old_from_new_[right] = temp;

      new_from_old_[temp] = right;
      new_from_old_[old_from_new_[left]] = left;

      left_vector.SwapValues(&right_vector);
      temp = (int)split[left - start];
      split[left - start] = split[right - start];
      split[right - start] = temp;
            
      // Update linked lists
      for (i = 0; i < n_features_; i++){
	if (order_[i].length() > 0){
	  int order_l, order_r, back_l, back_r;
	  order_l = (int)order_[i][left];
	  order_r = (int)order_[i][right];
	  back_l = (int)back_order_[i][left];
	  back_r = (int)back_order_[i][right];
	 
	  if (likely(order_l >= 0)){
	    back_order_[i][order_l] = right;	    
	  } 
	  order_[i][left] = order_r;
	  
	  if (likely(order_r >= 0)){
	    back_order_[i][order_r] = left;
	  }
	  order_[i][right] = order_l;

	  if (likely(back_l >= 0)){
	    order_[i][back_l] = right;	  
	  } else if (back_l == -1){
	    firsts_r[i] = right;
	  } 
	  back_order_[i][right] = back_l;
	  
	  if (likely(back_r >= 0)){
	    order_[i][back_r] = left;
	  } else if (back_r == -1){
	    firsts_l[i] = left;
	  }	 
	  back_order_[i][left] = back_r;	  
	}
      }      
      
      DEBUG_ASSERT(left <= right);
      right--;      
    }
    
    DEBUG_ASSERT(left == right + 1);
    firsts_l_out->Init(n_features_);
    firsts_r_out->Init(n_features_);
    firsts_l_out->CopyValues(firsts_l);
    firsts_r_out->CopyValues(firsts_r);
    
    return left;
  } //MatrixPartition
  


  int GetVariableType(int dim) {    
    DatasetFeature temp = data_.info().feature(dim);
    return temp.n_values();    
  }

  int GetFeatures(){
    return n_features_;   
  }

  int GetPointSize(){
    return n_points_;
  }

  int GetTargetType(int target_dim){
    if (target_dim < data_.n_features()){
      DatasetFeature temp = data_.info().feature(target_dim);
      return temp.n_values();
    } else {
      return target_types_;
    }
  }
 

  double Verify(int target_dim, double value, int index){
    double real_value = this->Get(target_dim, index);
    int target_type = this->GetTargetType(target_dim);
    if (target_type > 0){
      return !((int)value == (int)real_value);
    } else {
      return (value - real_value);
    }
    return 0;
  }

  void GetOrder(int dim, Vector *order, int start, int stop){
    order->WeakCopy(order_[dim]);    
  }

  double Get(int i, int j){
    return data_matrix_->get(i,j);
  }

  int WhereNow(int i){
    return new_from_old_[i];
  }

}; // class DataSet

#endif
