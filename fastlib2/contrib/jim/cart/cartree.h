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
  int start_, stop_, num_nodes_, split_point_;
  int target_dim_;
  double error_, complexity_cost_, child_error_;
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
	    int start_in, int stop_in, int target_dim_in, 
	    double complexity_cost_in) {
    points_ = points_in;
    start_ = start_in;
    stop_ = stop_in;
    complexity_cost_ = complexity_cost_in;
    left_ = NULL;
    right_ = NULL;
    first_pointers_.Copy(first_pointers_in);
    target_dim_ = target_dim_in;
    error_ = BIG_BAD_NUMBER;
  }

  void Init(TrainingSet* points_in, Vector &first_pointers_in, 
	    int start_in, int stop_in, int target_dim_in,
	    double complexity_cost_in, double error_in){
    points_ = points_in;   
    start_ = start_in;
    stop_ = stop_in;
    complexity_cost_ = complexity_cost_in;
    left_ = NULL;
    right_ = NULL;
    error_ = error_in;
    first_pointers_.Copy(first_pointers_in);
    target_dim_ = target_dim_in;
  }

  void Prune(double lambda){
    if (likely(left_ != NULL)){
      left_->Prune(lambda);
      right_->Prune(lambda);
      num_nodes_ = 1 + left_->GetNumNodes() + right_->GetNumNodes();
      child_error_ = left_->GetChildError() + right_->GetChildError();
      if (error_ - child_error_ < (lambda * (num_nodes_ ))) {
	left_ = NULL;
	right_ = NULL;
	child_error_ = error_;
	num_nodes_ = 1;
      }
    }
  }


  void Grow() {    
   
    if (error_ > complexity_cost_){
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
		  complexity_cost_, left_error);
      right_->Init(points_, first_pts_r, split_point_, stop_, target_dim_,
		   complexity_cost_, right_error);       
      left_->Grow();      
      right_->Grow();      
      child_error_ = left_->GetChildError() + right_->GetChildError();
      num_nodes_ = left_->GetNumNodes() + right_->GetNumNodes();
    } else {
      split_criterion_.InitStupid();
      child_error_ = error_;
      num_nodes_ = 1;
    }
  }
  
  double GetChildError(){
    return child_error_;
  }

  int GetNumNodes(){
    return num_nodes_;
  }
  
}; // Class CARTree

#endif
