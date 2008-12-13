/**
 * @file cartree.h
 *
 * Classification and regression Tree class
 *
 */

#ifndef CARTREE_H
#define CARTREE_H

#include "fastlib/fastlib.h"
#include "split_tree.h"

class CARTree{
 

  /////////////////// Nested Classes /////////////////////////////////////////


  ////////////////////// Member Variables /////////////////////////////////////
  
 private:
  int start_, stop_, split_point_;
  int target_dim_;
  double error_, value_;
  Split split_criterion_;
  CARTree *left_;
  CARTree *right_;
  TrainingSet *points_;
  Vector first_pointers_;
  int test_count_; 
  double test_error_;

  ////////////////////// Private Functions ////////////////////////////////////
  
  void ErrorInit_(){
    int targ_type = points_->GetTargetType(target_dim_);    
    int j, count = 0;
    double temp_err = 0.0;
    if (targ_type > 0) {
      int val, mode = 0;
      int temp_value = 0;
      Vector abundances;
      abundances.Init(targ_type);
      abundances.SetZero();
      int eta = 0;
      for (j = start_; j < stop_; j++){		
	val = (int)points_->Get(target_dim_, j);
	eta  = eta + 2*(int)abundances[val] + 1;
	count++;
	abundances[val] = abundances[val] + 1;
	if (abundances[val] > mode){
	  mode++;
	  temp_value = val;
	}
      }
      error_ = 1 - eta / (count*count);
      value_ = temp_value;
    } else {      
      double mean = 0.0;
      for (j = start_; j < stop_ ; j++){
	double val = points_->Get(target_dim_, j);
	temp_err = temp_err + count*(mean - val)*(mean - val) / (count + 1);
	mean = (mean*count + val) / (count + 1);
	count++;
      }
      error_ = sqrt(temp_err);
      value_ = mean;
    }   
  }


  ////////////////////// Constructors /////////////////////////////////////////
  
  FORBID_ACCIDENTAL_COPIES(CARTree);

 public: 

  CARTree() {
  }

  ~CARTree() {
    if (left_ != NULL){
      delete left_;
    }
    if (right_ != NULL){
      delete right_;
    }
  }

  ///////////////////// Helper Functions //////////////////////////////////////

  
  // Root node initializer
  void Init(TrainingSet* points_in, Vector &first_pointers_in, 
	    int start_in, int stop_in, int target_dim_in) {
    points_ = points_in;
    start_ = start_in;
    stop_ = stop_in;   
    left_ = NULL;
    right_ = NULL;
    first_pointers_.Copy(first_pointers_in);
    target_dim_ = target_dim_in;
    ErrorInit_();
  }

  void Init(TrainingSet* points_in, Vector &first_pointers_in, 
	    int start_in, int stop_in, int target_dim_in, double error_in){
    points_ = points_in;   
    start_ = start_in;
    stop_ = stop_in;   
    left_ = NULL;
    right_ = NULL;
    
    error_ = error_in / (stop_ - start_);
   
    first_pointers_.Copy(first_pointers_in);
    target_dim_ = target_dim_in;    
  }

  
 /*
   * Prune tree- remove subtrees if small decrease in
   * Gini index does not justify number of leaves.
   */
 
  double Prune(double lambda){
    if (likely(left_ != NULL)){
      double result;
      result = left_->Prune(lambda);
      result = min(result, right_->Prune(lambda));     
      double child_error;
      int leafs;
      leafs = left_->GetNumNodes() + right_->GetNumNodes();
      child_error = left_->GetChildError()*left_->Count()
	+ right_->GetChildError()*right_->Count();      
      double criterion = (stop_ - start_)*error_ - child_error;   
      if (criterion <= (lambda * (leafs-1))) {
	left_ = NULL;
	right_ = NULL;
	return BIG_BAD_NUMBER;
      } else {	
	return min(result, criterion / (leafs-1));
      }	
    } else {
      return BIG_BAD_NUMBER;
    }
  }
  

  void SetTestError(TrainingSet* test_data, int index){
    test_count_++;
    if (likely(left_ != NULL)){
      ArrayList<ArrayList<double> > split_data;
      split_data = split_criterion_.GetSplitParams();
      int split_dim = (int)split_data[0][0];
      // Split variable is not ordered...
      if (test_data->GetVariableType(split_dim) > 0 ){
	bool go_left = 0;
	for (int i = 1; i < split_data[0].size(); i++){
	  go_left = go_left | 
	  (split_data[0][i] == (int)(test_data->Get(split_dim, index)));
	}
	if (go_left){
	  left_->SetTestError(test_data, index);
	} else {
	  right_->SetTestError(test_data, index);
	}
      } else {
      // Split Variable is ordered.
 	double split_point = split_data[0][1];
	if (test_data->Get(split_dim, index) >= split_point){
	  right_->SetTestError(test_data, index);
	} else {
	  left_->SetTestError(test_data, index);
	}
      }
    } 
    if (test_data->GetTargetType(target_dim_) > 0) {
      test_error_ = test_error_ + 
	(value_ != test_data->Get(target_dim_, index));
    } else {
      test_error_ = test_error_ + 
	(value_ - test_data->Get(target_dim_, index))*
	(value_ - test_data->Get(target_dim_, index));
    }
  }

  double GetTestError(){
    if (likely(left_ != NULL)){
      return left_->GetTestError() + right_->GetTestError();
    } else {
      if (points_->GetTargetType(target_dim_) > 0) {
	return test_error_;
      } else {
	return sqrt(test_error_ / test_count_);
      }
    }
  }

  void WriteTree(int level, FILE* fp){
    if (likely(left_ != NULL)){
      fprintf(fp, "\n");
      for (int i = 0; i < level; i++){
	fprintf(fp, "|\t");
      }
      ArrayList<ArrayList<double> > split_data;
      split_data = split_criterion_.GetSplitParams();      
      int split_dim;
      double split_val;
      split_dim =  (int)split_data[0][0];
      split_val = split_data[0][1];      
      fprintf(fp, "Var. %d >=%5.2f ", split_dim, split_val);
      right_->WriteTree(level+1, fp);
      fprintf(fp, "\n");
      for (int i = 0; i < level; i++){
	fprintf(fp, "|\t");
      }      
      fprintf(fp, "Var. %d < %5.2f ", split_dim, split_val);
      left_->WriteTree(level+1, fp);
    } else {            
      fprintf(fp, ": Predict =%4.0f", value_);
    }  
  }
  
  /*
   * Find predicted value for a given data point
   */
  double Test(TrainingSet* test_data, int index){   
    if (likely(left_ != NULL)){
      ArrayList<ArrayList<double> > split_data;
      split_data = split_criterion_.GetSplitParams();
      int split_dim = (int)split_data[0][0];
      // Split variable is not ordered...
      if (test_data->GetVariableType(split_dim) > 0 ){
	bool go_left = 0;
	for (int i = 1; i < split_data[0].size(); i++){
	  go_left = go_left | 
	  (split_data[0][i] == (int)(test_data->Get(split_dim, index)));
	}
	if (go_left){
	  return left_->Test(test_data, index);
	} else {
	  return right_->Test(test_data, index);
	}
      } else {
      // Split Variable is ordered.
 	double split_point = split_data[0][1];
	if (test_data->Get(split_dim, index) >= split_point){
	  return right_->Test(test_data, index);
	} else {
	  return left_->Test(test_data, index);
	}
      }
    } else{
      return value_;
    }
  } // Test


  /*
   * Expand tree
   */
  void Grow() {    
    if (error_ > 1e-8 & start_ != stop_){
      split_criterion_.Init(points_, first_pointers_, start_, stop_, 
			    target_dim_);    
      Vector first_pts_l, first_pts_r;
      /*  
      printf("\n");
      printf("Start: %d Stop: %d Error: %f \n", start_, stop_, error_);
      printf("-------------------------------------------\n");
      */
      split_criterion_.MakeSplit(&first_pts_l, &first_pts_r);
      
      split_point_ = split_criterion_.GetSplitPoint();
      
      double left_error = split_criterion_.GetLeftError();
      double right_error = split_criterion_.GetRightError();
      //  printf("Left Error: %f Right Error: %f \n", left_error, right_error);
      // printf("Split Point: %d \n", split_point_);
      left_ = new CARTree();
      right_ = new CARTree();      
      left_->Init(points_, first_pts_l, start_, split_point_, target_dim_,
		  left_error);
      right_->Init(points_, first_pts_r, split_point_, stop_, target_dim_,
		   right_error);    
      left_->SetValue(split_criterion_.GetLeftValue());
      right_->SetValue(split_criterion_.GetRightValue());     
      left_->Grow();      
      right_->Grow();                
    } else {
      split_criterion_.InitStupid();          
    }
  } // Grow
  

  double GetChildError(){
    if (left_ != NULL){
      return (left_->GetChildError()*left_->Count() + 
	      right_->GetChildError()*right_->Count()) / (stop_ - start_);
    } else {
      return error_;
    }
  }

  int GetNumNodes(){
    if (left_  != NULL){
      return left_->GetNumNodes() + right_->GetNumNodes();
    } else {
      return 1; 
    }
  }

  int Count(){
    return stop_ - start_;
  }

  void SetValue(double val_in){
    value_ = val_in;
  }
  
}; // Class CARTree

#endif
