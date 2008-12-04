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


  ////////////////////// Private Functions ////////////////////////////////////
  
   

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
    error_ = BIG_BAD_NUMBER;
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
  void Prune(double lambda){
    if (likely(left_ != NULL)){
      left_->Prune(lambda);
      right_->Prune(lambda);     
      double child_error;
      int leafs;
      leafs = left_->GetNumNodes() + right_->GetNumNodes();
      child_error = left_->GetChildError()*left_->Count()
	+ right_->GetChildError()*right_->Count();       
      if ((stop_ - start_)*error_ <= child_error + (lambda * (leafs-1))) {
	//	printf("Pruning %d leaves from tree. \n", leafs);
	left_ = NULL;
	right_ = NULL;
      }
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
	  (split_data[0][i] == (int)(test_data->Get(index, split_dim)));
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
    if (error_ > 1e-8){
      split_criterion_.Init(points_, first_pointers_, start_, stop_, 
			    target_dim_);
      if (error_ == BIG_BAD_NUMBER){
	error_ = split_criterion_.GetInitialError();
      }
      Vector first_pts_l, first_pts_r;
      printf("\n");
      printf("Start: %d Stop: %d Error: %f \n", start_, stop_, error_);
      printf("-------------------------------------------\n");
      split_criterion_.MakeSplit(&first_pts_l, &first_pts_r);
      
      split_point_ = split_criterion_.GetSplitPoint();
      double left_error = split_criterion_.GetLeftError();
      double right_error = split_criterion_.GetRightError();
      printf("Left Error: %f Right Error: %f \n", left_error, right_error);
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
