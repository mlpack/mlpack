/*
 *  n_point_results.h
 *  
 *
 *  Created by William March on 4/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef N_POINT_RESULTS_H
#define N_POINT_RESULTS_H

#include "fastlib/fastlib.h"
#include "n_point_impl.h"

/**
 * Just a simple tensor class to handle results for the n-point problem
 */
class ResultsTensor {
  
private:
  
  int tensor_rank_;
  
  int num_bandwidths_;
  
  ArrayList<double> bandwidths_;
  
  // TODO: how is this organized?
  ArrayList<int> results_;
  
  //////////// functions //////////////////////
  
  index_t FindIndex_(const ArrayList<index_t>& indices);
  
public:
  
  void Init(int n, double min_band, double max_band, int num_bands) {
    
    DEBUG_ASSERT(max_band > min_band);
    DEBUG_ASSERT(num_bands > 0);
    
    tuple_size_ = n;
    
    num_bandwidths_ = num_bands;
    
    bandwidths_.Init(num_bandwidths_);
    
    double bandwidth_step = (max_band - min_band) / (double)num_bandwidths_;
    
    for (index_t i = 0; i < num_bandwidths_; i++) {
      
      bandwidths_[i] = min_band + (double)i * bandwidth_step;
      
    } // fill in bandwidths
    
    results_.Init(n_point_impl::NChooseR(num_bandwidths_ + tuple_size_ + 1, 
                                         tuple_size_));
    
  } // Init()
  
  int Get(const ArrayList<index_t>& indices) {
    
    index_t ind = FindIndex_(indices);    
    
    return results_[ind];
    
  } // get()
  
  void Set(const ArrayList<index_t>& indices, int val) {
    
    index_t ind = FindIndex_(indices);
    
    results_[ind] = val;
    
  } // set()

  void AddTo(const ArrayList<index_t>& indices, int val) {
    
    index_t ind = FindIndex_(indices);
    
    results_[ind] += val;
    
  } // AddTo()
  
  
  void SetRange(const ArrayList<index_t>& lower_ind, 
                const ArrayList<index_t>& upper_ind, int val);

  void AddToRange(const ArrayList<index_t>& lower_ind, 
                  const ArrayList<index_t>& upper_ind, int val);
  
  
  void Print() {
    
//    ArrayList<index_t> indices;
//    indices.InitRepeat(0, tuple_size_);
  
    // TODO: Make this format better
    for (index_t i = 0; i < results_.size(); i++) {

      printf("%d\n", results_[i]);
      
    } // initialize
    
    
    
  } // Print()
  
  
  
}; // NPointResults


#endif
