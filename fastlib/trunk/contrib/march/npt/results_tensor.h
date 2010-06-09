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
  int tuple_size_;
  
  int num_bins_;
  
  ArrayList<int> results_;
  int num_results_;
  
  ArrayList<bool> filled_results_;
  
  //////////// functions //////////////////////
  
  index_t FindIndex_(const ArrayList<index_t>& indices);
  
  bool IncrementIndex_(ArrayList<index_t>& new_ind, 
                       const ArrayList<index_t>& orig_ind, 
                       index_t k);
  
public:
  
  void Init(int n, int length) {
        
    tuple_size_ = n;
    tensor_rank_ = n_point_impl::NChooseR(n, 2);
    
    num_bins_ = length;
    
    num_results_ = 1;
    for (index_t i = 0; i < tensor_rank_; i++) {
      num_results_ *= num_bins_;
    }
    
    // The strictly upper triangular version
    //results_.InitRepeat(initial_result, 
    //                    n_point_impl::NChooseR(lengths_ + tuple_size_ + 1, 
    //                                           tuple_size_));
    results_.InitRepeat(0, num_results_);
    filled_results_.InitRepeat(false, num_results_);
    
  } // Init()
  
  int Get(const ArrayList<index_t>& indices) {
    
    index_t ind = FindIndex_(indices);    
    
    return results_[ind];
    
  } // get()
  
  void Set(const ArrayList<index_t>& indices, int val) {
    
    index_t ind = FindIndex_(indices);
    
    results_[ind] = val;
    
  } // set()
  
  void SetAll(int val) {
  
    results_.Clear();
    results_.InitRepeat(val, num_results_);
    
  }

  /*
  ArrayList<int>& results() const {
    return results_;
  }

  ArrayList<bool>& filled_results() const {
    return filled_results_;
  }
   */
  
  int tensor_rank() {
    return tensor_rank_;
  }
  
  int lengths() {
    return num_bins_;
  }
  
  void ClearFilledResults() {
    for (index_t i = 0; i < num_results_; i++) {
      filled_results_[i] = false;
    }
  }
  
  void IncrementRange(const GenMatrix<index_t>& lower_inds);
  
  
/*  
  void SetRange(const ArrayList<index_t>& lower_ind, 
                const ArrayList<index_t>& upper_ind, int val);

  void AddToRange(const ArrayList<index_t>& lower_ind, 
                  const ArrayList<index_t>& upper_ind, int val);
*/  
    
  /*
  void Copy(ResultsTensor& other) {
    
    results_.InitCopy(other.results());
    filled_results_.InitCopy(other.filled_results());
    
    tensor_rank_ = other.tensor_rank();
    lengths_ = other.lengths();
    
  } // Copy()
  */
  
  void Output(ArrayList<double>& distances_, FILE* fp) {
    
    ArrayList<index_t> indices;
    indices.InitRepeat(0, tensor_rank_);
    
    ArrayList<index_t> indices_copy;
    indices_copy.InitCopy(indices);
    
    bool done = false;
    
    while(!done) {
      
      //ot::Print(indices_copy);
      
      Matrix this_matcher;
      this_matcher.Init(tuple_size_, tuple_size_);
      
      index_t row_ind = 0;
      index_t col_ind = 1;
      
      for (index_t i = 0; i < indices.size(); i++) {

        double entry = sqrt(distances_[indices_copy[i]]);
        this_matcher.set(row_ind, col_ind, entry);
        this_matcher.set(col_ind, row_ind, entry);
        
        
        col_ind++;
        if (col_ind >= tuple_size_) {
          
          this_matcher.set(row_ind, row_ind, 0.0);
          row_ind++;
          col_ind = row_ind + 1;
          
        }
        
      } // fill in the matcher's matrix
      
      this_matcher.set(tuple_size_ - 1, tuple_size_ - 1, 0.0);
      
      index_t ind = FindIndex_(indices_copy);
      
      int this_result = results_[ind];
      
      // now do the printing
      
      this_matcher.PrintDebug("Matcher", fp);
      fprintf(fp, "==Result: %d==\n\n", this_result);
      
      
      done = IncrementIndex_(indices_copy, indices, 0);
      
    } // while
    
  } // Output()
  
  
}; // NPointResults


#endif
