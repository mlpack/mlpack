/*
 *  two_point.h
 *  
 *
 *  Created by William March on 2/16/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "fastlib/fastlib.h"

const fx_entry_doc naive_two_point_entries[] = {
{"total_runtime", FX_TIMER, FX_CUSTOM, NULL, 
"Total time required to compute the 2-point correlation.\n"},
{"radius", FX_REQUIRED, FX_DOUBLE, NULL,
  "The radius to compute the correlation for.\n"},
{"num_pairs", FX_RESULT, FX_INT, NULL,
  "The number of pairs found.\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc naive_two_point_submodules[] = {
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc naive_two_point_doc = {
naive_two_point_entries, naive_two_point_submodules,
"Algorithm module for naive serial two-point correlation.\n"
};


class NaiveTwoPoint {

private:
  
  Matrix data_points_;

  // TODO: replace with upper and lower bounds?
  // TODO: replace with a range of values and compute the correlation for all
  // of them?
  double radius_;
  
  int num_pairs_;
  index_t num_points_;
  
  fx_module* mod_;
  
public:
  
  void Init(const Matrix& data, fx_module* mod) {
    
    data_points_.Copy(data);
    mod_ = mod;
    
    num_points_ = data_points_.n_cols();
    
    radius_ = fx_param_double_req(mod_, "radius");
    
    if (radius_ <= 0.0) {
      FATAL("Negative radii not allowed.\n");
    }
    
    num_pairs_ = 0;
    
  } // Init()
  
  
  void Compute() {
    
    fx_timer_start(mod_, "total_runtime");

    for (index_t i = 0; i < num_points_ - 1; i++) {
      
      Vector i_vec;
      data_points_.MakeColumnVector(i, &i_vec);
      
      for (index_t j = i + 1; j < num_points_; j++) {
        
        Vector j_vec;
        data_points_.MakeColumnVector(j, &j_vec);
        
        double dist = sqrt(la::DistanceSqEuclidean(i_vec, j_vec));
        
        //printf("distance = %g\n", dist);
        
        if (dist < radius_) {
          
          num_pairs_++;
          
        } // is dist small
        
      } // for j
      
    } // for i
    
    fx_timer_stop(mod_, "total_runtime");
    
    fx_result_int(mod_, "num_pairs", num_pairs_);
    
    printf("Number of pairs: %d\n\n", num_pairs_);
    
  } // Compute()
  
  
}; // NaiveTwoPoint