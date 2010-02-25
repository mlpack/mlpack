/*
 *  n_point.h
 *  
 *
 *  Created by William March on 2/24/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_H
#define N_POINT_H

#include "matcher.h"
#include "fastlib/fastlib.h"

const fx_entry_doc n_point_entries[] = {
{"total_runtime", FX_TIMER, FX_CUSTOM, NULL, 
"Total time required to compute the n-point correlation.\n"},
{"n_point_time", FX_TIMER, FX_CUSTOM, NULL, 
"Time required for just the n-point computation (after tree building).\n"},
{"num_tuples", FX_RESULT, FX_INT, NULL,
"The number of matching tuples found.\n"},
{"weighted_num_tuples", FX_RESULT, FX_DOUBLE, NULL,
"The sum of the product of the weights over all matching tuples found.\n"},
{"do_naive", FX_PARAM, FX_BOOL, NULL,
 "If true, the algorithm just runs the base case on the entire data set.\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc n_point_submodules[] = {
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc n_point_doc = {
n_point_entries, n_point_submodules,
"Algorithm module for n-point correlation.\n"
};


/**
 *
 */
class NPointAlg {

private:
  
  Matrix data_points_;
  Vector data_weights_;
  
  int num_points_;
  int tuple_size_;
  
  Matcher matcher_;
  
  fx_module* mod_;
  
  bool do_naive_;
  
  int num_tuples_;
  double weighted_num_tuples_;
  ////////////// functions ////////////
  
  /**
   *
   */
  bool PointsViolateSymmetry_(index_t ind1, index_t ind2);
  
  
  
  /**
   *
   */
  int BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                                 ArrayList<bool>& permutation_ok,
                                 ArrayList<index_t>& points_in_tuple,
                                 double* weighed_result, int k);
  
  
  /**
   *
   */
  int BaseCase_(ArrayList<ArrayList<index_t> >& point_sets, 
                   double* weighted_result);
  
  
public:
  
  /**
   *
   */
  void Init(const Matrix& data, const Vector& weights, const Matrix& lower_bds, 
            const Matrix& upper_bds, int n, fx_module* mod) {
    
    mod_ = mod;
    
    data_points_.Copy(data);
    data_weights_.Copy(weights);
    
    num_points_ = data_points_.n_cols();
    
    tuple_size_ = n;
    
    matcher_.Init(lower_bds, upper_bds, tuple_size_);
    
    do_naive_ = fx_param_bool(mod_, "do_naive", true);
    
    num_tuples_ = 0;
    weighted_num_tuples_ = 0.0;
    
  } // Init()
  
  /**
   *
   */
  void Compute() {
    
    fx_timer_start(mod_, "n_point_time");
    
    if (do_naive_) {
      
      ArrayList<ArrayList<index_t> > point_sets;
      point_sets.Init(tuple_size_);
      for (index_t i = 0; i < tuple_size_; i++) {
        point_sets[i].Init(num_points_);
        
        for (index_t j = 0; j < num_points_; j++) {
          point_sets[i][j] = j;
        } // for j
        
      } // for i
      
      num_tuples_ = BaseCase_(point_sets, &weighted_num_tuples_);
      
    } // do naive
    else {
      
      FATAL("Multi tree version not yet implemented.\n");
      
    } // do multi-tree

    fx_timer_stop(mod_, "n_point_time");

    printf("\n====  Number of tuples: %d ====\n\n", num_tuples_);
    
    fx_result_int(mod_, "num_tuples", num_tuples_);
    fx_result_double(mod_, "weighted_num_tuples", weighted_num_tuples_);
    
  } // Compute()
  
  
}; // NPointAlg


#endif 

