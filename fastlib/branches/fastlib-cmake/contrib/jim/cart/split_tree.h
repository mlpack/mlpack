#ifndef SPLIT_TREE_H
#define SPLIT_TREE_H

#include "fastlib/fastlib.h"
#include "training_set.h"

class Split{

  ////////////////////// Member Variables /////////////////////////////////////
  
 private:
  TrainingSet* points_;
  int start_, stop_;
  int split_point_, split_dim_, target_dim_;
  double left_error_, right_error_;
  double trial_left_error_, trial_right_error_;
  double left_value_, right_value_;
  double trial_left_value_, trial_right_value_;
  Vector trial_split_, split_;
  Vector first_pointers_;
  ArrayList<ArrayList <double> > split_params_;
  ArrayList<double> trial_split_params_;
  int trial_missing_data_flag_, missing_data_flag_;
  

  ////////////////////// Private Functions ////////////////////////////////////
  
  // ArrayList of 3D vector respresents results for each class. 
  // First index of each vector is total points, second is mean,
  // and third is sum of squares dev.
  void NonOrderedRegression_(int dim, int classes) {
    trial_missing_data_flag_ = 0;
    ArrayList<Vector> results;
    trial_split_params_.Renew();    
    results.Init(classes);
    int i, j;
    
    // Obtain vector of count, average, and squared error for each 
    // variable class.
    for (i = 0; i < classes; i++){
      results[i].Init(4);
      results[i].SetZero();
      results[i][3] = i;
    }   
    for (i = start_; i < stop_; i++){
      j = (int)points_->Get(dim, i);
      if (! isnan(j)){
	double val = points_->Get(target_dim_, i);
	results[j][2] = results[j][2] + (results[j][0] / (results[j][0] + 1))*
	  (val - results[j][1])*(val - results[j][1]);    
	results[j][1] = (results[j][0] * results[j][1] + val) 
	  / (results[j][0] + 1);
	results[j][0] = results[j][0] + 1;      
      } else {
	trial_split_[i] = -1;
	trial_missing_data_flag_ = 1;
      }
    }

    // Find best partition of classes ...
    // Begin by sorting by mean. The number of classes should be 
    // be fairly small, so using an O(n^2) sorting method shouldn't
    // hurt too much.
    Vector left, right;   
    right.Init(3);
    right.SetZero();
    left.Init(4);
    for (i = 0; i < classes; i++){
      left.CopyValues(results[i]);
      
      for (j = i+1; j < classes; j++) {
	if (results[j][1] < left[1]) {
	  results[i].CopyValues(results[j]);
	  results[j].CopyValues(left);
	  left.CopyValues(results[i]);
	}
      }
      right[2] = right[2] + left[2] + left[0]*right[0] * 
	(left[1] - right[1])*(left[1] - right[1]) / (left[0] + right[0]);
      right[1] = (right[0] * right[1] + left[0] * left[1]) /  
	(left[0] + right[0]);
      if (left[0] + right[0] == 0){
	right[1] = 0;
	right[2] = 0;
      }
      right[0] = right[0] + left[0];   
    }
    left.SetZero();
    
    // Now move the boundary across the mean values
    // of the results for each variable class.
    double delta_err_r = 1.0, delta_err_l = 0.0;
    i = classes;
    while (delta_err_l <= delta_err_r && right[0] > 0){     
      i = i-1;      
      left[3] = results[i][3];

      // Update Error
      delta_err_r = results[i][2] + (right[0]*results[i][0])*
	(right[1] - results[i][1])*(right[1] - results[i][1]) / 
	(right[0] - results[i][0]);
      if (right[0] - results[i][0] == 0){
	delta_err_r = right[2];
      }
      delta_err_l = results[i][2] + (left[0]*results[i][0])*
	(left[1] - results[i][1])*(left[1] - results[i][1]) / 
	(left[0] + results[i][0]);
      left[2] = left[2] + delta_err_l;
      right[2] = right[2] - delta_err_r;

      // Update Means
      left[1] = (left[1] * left[0] + results[i][1] * results[i][0]) / 
	(left[0] + results[i][0]);
      right[1] = (right[1] * right[0] - results[i][1] * results[i][0]) / 
	(right[0] - results[i][0]);

      // Update tallys
      left[0] = left[0] + results[i][0];
      right[0] = right[0] - results[i][0];
      if (right[0] == 0){
	right[1] = 0;
	right[2] = 0;
      }
      if (left[0] == 0){
	left[1] = 0;
	left[2] = 0;
      }     
    }
    trial_split_params_.Init(classes - i);
    trial_split_params_[0] = dim;
    for (j = 1; j < classes - i; j++){
      trial_split_params_[j] = results[classes - j][3];
    }
    // Undo last move
    trial_left_error_ = left[2] - delta_err_l;
    trial_right_error_ = right[2] + delta_err_r;    
    
    trial_right_value_ = (right[0]*right[1] + results[i][0]* results[i][1]) / 
      (right[0] + results[i][0]);
    trial_left_value_ = (left[0]*left[1] - results[i][0]*results[i][1]) / 
      (left[0] - results[i][0]);   
    
    
    // Straighten out list of classes
    double cutoff_mean = results[i][1];    
    j = 0;
    while (j < classes){     
      int temp = (int)results[j][3];
      if (temp != j){
	left.CopyValues(results[j]);
	results[j].CopyValues(results[temp]);
	results[temp].CopyValues(left);		
      } else {
	j++;
      }  
    }    

    // Set index of vector split 1 / 0 to indeicate which side of 
    // split it lies on.
    for (j = start_; j < stop_; j++){
      i = (int)points_->Get(dim, j);
      if (results[i][1] > cutoff_mean) {
	trial_split_[j-start_] = 1;
      } else {
	trial_split_[j-start_] = 0;
      }
    }   
    
  }

  //Classify based on nominal variabale
  void NonOrderedClassification_(int dim, int v_classes, int t_classes){
    ArrayList<Vector> results;
    Vector count;
    Vector gini;
    Vector which_side;
    which_side.Init(v_classes);
    gini.Init(v_classes);
    count.Init(v_classes);
    results.Init(v_classes);
    gini.SetZero();
    count.SetZero();
    int i,j;
    for (i = 0; i < v_classes; i++){
      results[i].Init(t_classes);
      results[i].SetZero();
    }

    int val;
    for (i = start_; i < stop_; i++){
      if (target_dim_ >= 0){
      	j = (int)points_->Get(dim, i);
	val = (int)points_->Get(target_dim_, i);
      } else {
	val = -1;
	while (val < 0){
	  val = (int)split_[i - start_]; 
	  i++;
	}  
	i--;
	j = (int)points_->Get(dim, i);
      }
      // Update Gini error (Note this is actually N * Gini Error)
      gini[j] = gini[j] + 2*count[j] - 2*results[j][val];
      // Update tally for this target class, and for total instances of
      // this variable class.
      count[j] = count[j] + 1;
      results[j][val] = results[j][val] + 1;      
    }  
    // Now find best partition of variable classes...
    Vector left, right;
    right.Init(t_classes);
    left.Init(t_classes);
    right.SetZero();
    left.SetZero();
    if (v_classes > 2){
      int left_count = 0, right_count = 0;
      double left_gini = 0, right_gini = 0;  
      which_side.SetZero();             
      for (i = 0; i < v_classes; i++){
	right_gini = right_gini + gini[i] + 2 * (right_count * count[i] - 
				       la::Dot(right, results[i]));
	// Update Gini terms
	la::AddTo(results[i], &right);
	right_count = right_count + (int)count[i];
      } 

      // Attempt to move one feature class from right to left
      double best_change = -1e-5;
      int change_index;
      double change = 0;
      while(best_change  <= 0){
	change_index = -1;
	best_change = 1e-5;
	for (i = 0; i < v_classes; i++){
	  if (which_side[i] == 0){
	    // Try to move across cut
	    change = -(la::Dot(left, left) + 2*la::Dot(left, results[i]) + 
		      la::Dot(results[i], results[i]))/(left_count + count[i])
	      -(la::Dot(right, right) + la::Dot(results[i], results[i]) - 
		 2*la::Dot(results[i], right)) / (right_count - count[i]) + 
	      la::Dot(right, right) / right_count;
	    if (left_count > 0){
	      change = change + la::Dot(left, left)/ left_count; 
	    }	  
	    if (change < best_change){
	      best_change = change;
	      change_index = i;
	    }
	  }
	}
	// Commit Change
	if (change_index >= 0){	
	  which_side[change_index] = 1;
	  left_gini = left_gini + gini[change_index] + 2*
	    (count[change_index]*left_count - 
	     la::Dot(left, results[change_index]));	  
	  la::SubFrom(results[change_index], &right);
	  la::AddTo(results[change_index], &left);
	  left_count = left_count + (int)count[change_index];
	  right_count = right_count - (int)count[change_index];	
	  right_gini = right_gini - gini[change_index] - 2*
	    (count[change_index]*right_count - 
	     la::Dot(right, results[change_index]));	  
	}
      }
      trial_left_error_ = left_gini / left_count;
      trial_right_error_ = right_gini / right_count;
      
    } else {
      trial_left_error_ = results[0][1] / results[0][0];
      trial_right_error_ = results[1][1] / results[1][0];
      
      which_side[0] = 0;
      which_side[1] = 1;
    }   
    // Assign indices of split
    // Identify modes, flip 1 / 0 in which_side
    trial_left_value_ = 0;
    trial_right_value_ = 0;
    for (j = 0; j < t_classes; j++){
      if (left[j] > left[(int)trial_left_value_]){
	trial_left_value_ = j;
      }
      if (right[j] >= right[(int)trial_right_value_]){
	trial_right_value_ = j;
      }
    }
    int flip = 0;
    if (trial_right_value_ < trial_left_value_){
      flip = 1;
      double temp = trial_right_value_;
      trial_right_value_ = trial_left_value_;
      trial_left_value_ = (int)temp;
      temp = trial_right_error_;
      trial_right_error_ = trial_left_error_;
      trial_left_error_ = temp;
    }

    for (j = start_; j < stop_; j++){
      val = (int)points_->Get(dim, j);
      trial_split_[j - start_] = ((int)which_side[val])^flip;     
    }
  }

  // Predict value based upon ordered variable
  void OrderedRegression_(int dim) {
    trial_left_error_ = BIG_BAD_NUMBER;
    trial_missing_data_flag_ = 0;
    trial_split_params_.Renew();
    trial_split_params_.Init(2);
    trial_split_params_[0] = dim;
    int j, first;
    Vector order;
    points_->GetOrder(dim, &order, start_, stop_);
    first = (int)first_pointers_[dim];
    int current = first;

    // Get intial error, mean
    double left_mean = 0.0, right_mean = 0.0;
    double temp_right_error_ = 0.0, temp_left_error_ = 0.0;
    int right_points = 0, left_points = 0;
    int best_index = first;
    for (j = start_; j < stop_; j++){      
      
      double val = points_->Get(target_dim_, j);  
      if (!isnan(points_->Get(dim, j))){
	right_points++;
	trial_split_[j - start_] = 0;
	temp_right_error_ = temp_right_error_ + (right_points-1) *
	  (right_mean - val)*(right_mean - val) / right_points;
	right_mean = ((right_points - 1)*right_mean + val) / 
	  right_points;           
      } else {
	trial_split_[j - start_] = -1;
	trial_missing_data_flag_ = 1;
	printf("MISSING VALUE! Dimension %d \n", dim);
      }
    }   
    double best_error = BIG_BAD_NUMBER;

    // Move boundary across ordered values, one at a time.   
    while (current != -1 && right_points > 0) {  
      double val = points_->Get(target_dim_, current);
      temp_right_error_ = temp_right_error_ - right_points*
	(right_mean - val)*(right_mean - val) / (right_points - 1);
      temp_left_error_ = temp_left_error_ + left_points*
	(left_mean - val)*(left_mean - val) / (left_points + 1);
      
      right_mean = (right_points * right_mean - val) / (right_points - 1);
      left_mean = (left_points * left_mean + val) / (left_points + 1);
      right_points--;
      left_points++;
      if (right_points == 0){
	temp_right_error_ = 0;
	right_mean = 0;
      }     
      double current_var, next_var = BIG_BAD_NUMBER;
      current_var = points_->Get(dim, current);
      current = (int)order[current];
      double net_error = sqrt(temp_left_error_)*left_points + 
	sqrt(temp_right_error_)*right_points;      
      if (current != -1){
	next_var = points_->Get(dim, current);
      }
     
      // Keep track of lowest sum-of-squares error so far
      if (net_error < best_error & current_var != next_var){
	trial_split_params_[1] = points_->Get(dim, current);
	trial_right_value_ = right_mean;
	trial_left_value_ = left_mean;
	trial_right_error_ = sqrt(temp_right_error_)*right_points;
	trial_left_error_ = sqrt(temp_left_error_)*left_points;
	best_error = trial_right_error_ + trial_left_error_;
	int temp = best_index;
	while(temp != current) {	  
	  trial_split_[temp - start_] = 1;
	  temp = (int)order[temp];	  
	}
	best_index = current;
      }
    }     
  }  // OrderedRegression


  // Predict nominal target value based upon ordered variable.
  // A vector of length equal to the number of target classes
  // stores the numbe rof occurneces of each. Additional variables
  // track the most common target value, and the Gini error.
  void OrderedClassification_(int dim, int classes){        
    trial_left_error_ = BIG_BAD_NUMBER;
    trial_missing_data_flag_ = 0;
    trial_split_params_.Renew();
    trial_split_params_.Init(2);
    trial_split_params_[0] = dim;
    int j, first;
    Vector order;
    points_->GetOrder(dim, &order, start_, stop_);
    first = (int)first_pointers_[dim];
    int current = first;

    // Get intial error, mean
    int left_mode = 0, right_mode = 0;
    double temp_right_error_ = 0.0, temp_left_error_ = 0.0;
    int right_points = 0, left_points = 0;
    Vector left_abundance, right_abundance;
    left_abundance.Init(classes);
    left_abundance.SetZero();
    right_abundance.Init(classes);
    right_abundance.SetZero();
    int val;  
    int best_index = first;
    for (j = start_; j < stop_; j++){      
      if (target_dim_ >= 0 ){
	val = (int)points_->Get(target_dim_, j);
      } else {
	val = -1;
	while (val < 0){
	  val = (int)split_[j - start_];
	  j++;
	}
	j--;
      }
      if (!isnan(points_->Get(dim, j))){	
	trial_split_[j -start_] = 0;	
	temp_right_error_ = temp_right_error_ + 1 + 
	  2*(int)right_abundance[val];
	right_abundance[val] = right_abundance[val] + 1;
	if (right_abundance[val] > right_abundance[right_mode]){
	  right_mode = val;
	}
	right_points++;      
      } else {
	trial_split_[j - start_] = -1;
	trial_missing_data_flag_ = 1;
	printf("MISSING VALUE! \n");
      }
    }     
    double best_error = temp_right_error_ / right_points;   
    //printf("Initial Error: %f Dimension: %d \n", best_error,  dim);   
    while (current != -1) {
      int val = (int)points_->Get(target_dim_, current);
      temp_right_error_ = temp_right_error_ + 1 - 
	2 * (int)right_abundance[val]; 
      temp_left_error_ = temp_left_error_ + 1 + 2 * (int)left_abundance[val];  
      right_points--;
      if (right_points == 0){
	temp_right_error_ = 0;
      }
      left_points++;
      right_abundance[val] = right_abundance[val] - 1;
      left_abundance[val] = left_abundance[val] + 1;
      if (val == right_mode){
	for (j = 0; j < classes; j++){
	  if (right_abundance[j] > right_abundance[val]){
	    right_mode = j;
	  }
	}
      } 
      if (left_abundance[val] > left_abundance[left_mode]){
	left_mode = val;
      }
      double old_variable = points_->Get(dim, current);
      current = (int)order[current];
      if (current >= 0 && old_variable != points_->Get(dim, current)){
	if (right_points > 0 && temp_left_error_ / left_points + 
	    temp_right_error_ / right_points  > best_error){
	  trial_split_params_[1] = points_->Get(dim, current);
	  trial_right_value_ = right_mode;
	  trial_left_value_ = left_mode;
	  trial_right_error_ = right_points - temp_right_error_ / right_points;
	  trial_left_error_ = left_points - temp_left_error_ / left_points;
	  best_error = temp_right_error_ / right_points + 
	    temp_left_error_  / left_points;
	  int temp = best_index;
	  while(temp != current) {	    
	    trial_split_[temp - start_] = 1;	 
	    temp = (int)(order)[temp];
	  }
	  best_index = current;
	}
      }
    }
  } // OrderedClassification


  /*
   * Under construction- this will handle input files with missing data
   */
  /*
  void SurrogateSplits_(){
    int old_target_ = target_dim_;
    int i, n;
    double current_error;
    target_dim_ = -1;
    for (i = 0; i < n; i++) {
      if (likely(i != split_dim_ & i != old_target_)) {
	int var_type = points_->GetVariableType(i);
	if (var_type > 0){
	  NonOrderedClassification_(i, var_type, 2);
	} else {
	  OrderedClassification_(i, 2);
	}	
	// Place trial split in sequence
	current_error = trial_right_error_ + trial_left_error_;	
      } 
    } 
    // Update split for points with missing values
    for (i = start_; i <stop_; i++){
      int j = 1;
      while (split_[i - start_] == -1){
	ArrayList<double> next_best_split = split_params_[j];
	if (!isnan(points_->Get((int)next_best_split[0], i))){
	  // Nominal
	  if (points_->GetVariableType((int)next_best_split[0]) > 0){
	    split_[i - start_] = 0;
	    int k;
	    for (k = 1; k < next_best_split.size(); k++){
	      if (next_best_split[k] == 
		  points_->Get((int)next_best_split[0], i)){
		split_[i - start_] = 1;
	      }
	    }
	  } else {
	  //Ordered
	    if (points_->Get((int)next_best_split[0], i) < next_best_split[1]){
	      split_[i - start_] = 1;
	    } else{
	      split_[i - start_] = 0;
	    }
	  }
	}
	j++;
      }
    }
  } */
 

  ////////////////////// Constructors /////////////////////////////////////////
  
  FORBID_ACCIDENTAL_COPIES(Split);

 public: 

  Split() {
  }

  ~Split() {
  }

  ///////////////////// Helper Functions //////////////////////////////////////

  void Init(TrainingSet* points_in, Vector first_pointers_in, int start_in, 
	    int stop_in, int target_dim_in) {
    points_ = points_in;
    start_ = start_in;
    stop_ = stop_in;    
    trial_split_.Init(stop_ - start_);
    split_.Init(stop_ - start_);
    first_pointers_.Copy(first_pointers_in);
    target_dim_ =  target_dim_in;       
    split_params_.Init(1);
    split_params_[0].Init(0);
    trial_split_params_.Init(0);
  }

  void InitStupid(){
    points_ = NULL;
    split_.Init(1);
    trial_split_.Init(1);
    first_pointers_.Init(1);
    split_params_.Init(0);
    trial_split_params_.Init(0);
  }

  double GetInitialError(){
    int targ_type = points_->GetTargetType(target_dim_);    
    int j, count = 0;
    double error = 0.0;
    if (targ_type > 0) {
      int val;
      Vector abundances;
      abundances.Init(targ_type);
      abundances.SetZero();
      int eta = 0;
      for (j = start_; j < stop_; j++){		
	val = (int)points_->Get(target_dim_, j);
	eta  = eta + 2*(int)abundances[val] + 1;
	count++;
	abundances[val] = abundances[val] + 1;
      }
      error = 1.0 - eta / (count*count);
    } else {      
      double mean = 0.0;
      for (j = start_; j < stop_ ; j++){
	double val = points_->Get(target_dim_, j);
	error = error + count*(mean - val)*(mean - val) / (count + 1);
	mean = (mean*count + val) / (count + 1);
	count++;
      }
    }
    return error;
  }

  void MakeSplit(Vector* first_points_l, Vector* first_points_r) {    
    missing_data_flag_ = 0;
    int i, n = points_->GetFeatures();  
    left_error_ = BIG_BAD_NUMBER;
    right_error_ = BIG_BAD_NUMBER;   
    for (i = 0; i < n; i++){      
      int targ_type = points_->GetTargetType(target_dim_);
      if (i != target_dim_){
	int var_type = points_->GetVariableType(i);
	if (target_dim_ != i){
	  if (var_type > 0){ 
	    if (targ_type > 0){
	      NonOrderedClassification_(i, var_type, targ_type);
	    } else {
	      NonOrderedRegression_(i, var_type);
	    }	
	  } else {
	    if (targ_type > 0){
	      OrderedClassification_(i, targ_type);
	    } else {
	      OrderedRegression_(i);
	    }	
	  }	   
	  if (unlikely(trial_left_error_ + trial_right_error_ < 
		       left_error_ + right_error_)){	 
	    split_params_[0].Renew();
	    split_params_[0].InitCopy(trial_split_params_);
	    left_error_ = trial_left_error_;
	    right_error_ = trial_right_error_;	   
	    left_value_ = trial_left_value_;
	    right_value_ = trial_right_value_;
	    split_.CopyValues(trial_split_);
	    missing_data_flag_ = trial_missing_data_flag_;	
	    split_dim_ = i;
	  }      
	}
      }
    }

    /*
    // Surrogate splits
    if (missing_data_flag_){
      SurrogateSplits_();
      }*/

    /*
    printf("Split Variable: %d Value:", split_dim_);
    for (i = 1; i < split_params_[0].size(); i++){
      printf(" %5.3f, ", split_params_[0][i]);
    }
    printf("\n");
    printf("Left Value: %f Right value: %f \n", left_value_, right_value_);
    */
      
    // This will be the last line of the function;
    // printf("Partitioning Matrx ... \n");
    split_point_ = points_->MatrixPartition(start_, stop_, 
					    split_, first_pointers_,
					    first_points_l, first_points_r);
    //   printf("Matrix Partitioned. Split Point: %d \n", split_point_);
  }

  int GetSplitPoint() {
    return split_point_;
  }

  double GetRightError(){
    //  if (points_->GetTargetType(target_dim_) > 0){      
    // } else {
    return right_error_;
    //  }      
  }

  double GetLeftError() {
    return left_error_;
  }

  double GetLeftValue(){
    return left_value_;
  }

  double GetRightValue(){
    return right_value_;
  }

  ArrayList<ArrayList<double> > GetSplitParams(){
    return split_params_;
  }


}; // class Split

#endif
