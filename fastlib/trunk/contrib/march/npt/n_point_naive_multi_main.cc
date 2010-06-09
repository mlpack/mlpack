/*
 *  n_point_naive_multi_main.cc
 *  
 *
 *  Created by William March on 6/9/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#include "fastlib/fastlib.h"
#include "n_point_perm_free.h"
#include "n_point.h"

bool IncIndex(ArrayList<index_t>& new_ind, 
              const ArrayList<index_t>& orig_ind, 
              index_t k, int tensor_rank_, int num_bins_) {
  
  if (k >= tensor_rank_) {
    return true;
  }
  
  new_ind[k]++;
  if (new_ind[k] >= num_bins_) {
    new_ind[k] = orig_ind[k];
    return IncIndex(new_ind, orig_ind, k+1, tensor_rank_, num_bins_);
  }
  else {
    return false;
  }
  
} // IncrementIndex

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
  
  int n = fx_param_int_req(NULL, "n");
  int n_choose_2 = n_point_impl::NChooseR(n, 2);
  
  Matrix data;
  const char* data_file = fx_param_str_req(NULL, "data");
  data::Load(data_file, &data);
  
  Matrix lower_bounds;
  lower_bounds.Init(n, n);
  lower_bounds.SetAll(0.0);
  
  double min_band = fx_param_double_req(NULL, "min_band");
  double max_band = fx_param_double_req(NULL, "max_band");
  double num_bands = fx_param_int_req(NULL, "num_bands");
  
  ArrayList<double> dists;
  dists.Init(num_bands);
  
  
  double this_dist = min_band;
  double dist_step = (max_band - min_band) / (double)(num_bands-1);
  
  //printf("dist_step = %g\n", dist_step);
  
  // TODO: double check this
  for (index_t i = 0; i < num_bands; i++) {
    
    dists[i] = this_dist;
    this_dist += dist_step;
    
  }
  
  ArrayList<index_t> indices;
  indices.InitRepeat(0, n_choose_2);
  
  ArrayList<index_t> indices_copy;
  indices_copy.InitCopy(indices);

  
  // form each matcher
  
  bool done = false;
  
  fx_timer_start(NULL, "n_point_time");
  
  while (!done) {
    
    Matrix this_matcher;
    this_matcher.Init(n, n);
    
    index_t row_ind = 0;
    index_t col_ind = 1;
    
    for (index_t i = 0; i < indices.size(); i++) {
      
      double entry = dists[indices_copy[i]];
      this_matcher.set(row_ind, col_ind, entry);
      this_matcher.set(col_ind, row_ind, entry);
      
      col_ind++;
      if (col_ind >= n) {
        
        this_matcher.set(row_ind, row_ind, 0.0);
        row_ind++;
        col_ind = row_ind + 1;
        
      }
      
    } // fill in the matcher's matrix
    
    this_matcher.set(n - 1, n - 1, 0.0);
    
    done = IncIndex(indices_copy, indices, 0, n_choose_2, num_bands);
    
    // make the alg, run it
    NPointPermFree perm_free_alg;
    fx_module* submod = fx_submodule(NULL, "perm_free_mod");
    
    //this_matcher.PrintDebug("This matcher");
    perm_free_alg.Init(data, lower_bounds, this_matcher, n, submod);
    
    perm_free_alg.Compute();
    
  } // while
  
  fx_timer_stop(NULL, "n_point_time");
  
  fx_done(NULL);
  
  return 0;
                              
} // main