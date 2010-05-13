/*
 *  n_point_perm_free.h
 *  
 *
 *  Created by William March on 4/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_PERM_FREE_H
#define N_POINT_PERM_FREE_H

#include "perm_free_matcher.h"
#include "n_point_nodes.h"


class NPointPermFree {
  
private:
  
  Matrix data_points_;
  
  PermFreeMatcher matcher_;
  
  int tuple_size_;
  
  int num_tuples_;
  
  fx_module* mod_;
  
  NPointNode* tree_;
  
  int leaf_size_;
  
  int num_exclusion_prunes_;
  
  ////////////// functions ///////////////////
  
  bool PointsViolateSymmetry_(index_t ind1, index_t ind2);

  int BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                      ArrayList<bool>& permutation_ok,
                      ArrayList<index_t>& points_in_tuple, int k);  
  
  int BaseCase_(NodeTuple& nodes);

  bool CanPrune_(NodeTuple& nodes);

  int DepthFirstRecursion_(NodeTuple& nodes);
  
public:
  
  void Init(const Matrix& data, const Matrix& lower_bds, 
            const Matrix& upper_bds, int n, fx_module* mod) {
    
    mod_ = mod;
    
    data_points_.Copy(data);
    
    tuple_size_ = n;
    
    ArrayList<double> upper_dists;
    upper_dists.Init();
    for (index_t i = 0; i < tuple_size_; i++) {
      for (index_t j = i+1; j < tuple_size_; j++) {
        upper_dists.PushBackCopy(upper_bds.get(i,j));
      }
    }
    
    matcher_.Init(upper_bds, upper_dists, tuple_size_);
    
    leaf_size_ = fx_param_int(mod_, "leaf_size", 1);
    
    if (leaf_size_ <= 0) {
      FATAL("Leaf size must be strictly positive.\n");
    }
    
    
    num_tuples_ = 0;
    
    // why won't this compile without the old indices?
    ArrayList<index_t> old_from_new;
    tree_ = tree::MakeKdTreeMidpoint<NPointNode> (data_points_, leaf_size_,
                                                  &old_from_new, NULL);
    
    //tree_->Print();
    
    num_exclusion_prunes_ = 0;
    
  } // Init()
  
  
  void Compute() {
    
    fx_timer_start(mod_, "n_point_time");

    // make node tuple
  
    NodeTuple nodes;
    ArrayList<NPointNode*> node_list;
    node_list.Init(tuple_size_);
    
    
    for (index_t i = 0; i < tuple_size_; i++) {
         
      node_list[i] = tree_;
      
    } // for i
    
    nodes.Init(node_list);
    
    num_tuples_ = DepthFirstRecursion_(nodes);
    
    
    
    fx_timer_stop(mod_, "n_point_time");

    
    printf("\n====  Number of tuples: %d ====\n\n", num_tuples_);
    
    fx_result_int(mod_, "num_tuples", num_tuples_);
    fx_result_int(mod_, "num_exclusion_prunes", num_exclusion_prunes_);
    
    
  } // Compute()
  
  
}; // NPointPermFree





#endif