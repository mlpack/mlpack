#ifndef SUM_H
#define SUM_H

#include "fastlib/fastlib.h"
#include "operator.h"

class Sum: public Operator {

 public:

  double NaiveCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double sum_result = 0;

    // If this is the base case, where there is no more nested
    // operator inside, then call the multi-tree algorithm.
    if(operators_.size() > 0) {
      
    }

    // Otherwise, recursively evaluate the operators and add them up.
    else {
      
      for(index_t i = 0; i < operators_.size(); i++) {
	sum_result += operators_[i]->NaiveCompute();
      }
    }
    
    if(!is_positive_) {
      sum_result = -sum_result;
    }
    if(should_be_inverted_) {
      sum_result = 1.0 / sum_result;
    }
    
    return sum_result;
  }

  double MonteCarloCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double sum_result = 0;

    // If this is the base case, where there is no more nested
    // operator inside, then call the multi-tree algorithm.
    if(operators_.size() == 0) {
      
    }

    // Otherwise, recursively evaluate the operators and add them up.
    else {
      
      for(index_t i = 0; i < operators_.size(); i++) {
	sum_result += operators_[i]->MonteCarloCompute();
      }
    }
    
    if(!is_positive_) {
      sum_result = -sum_result;
    }
    if(should_be_inverted_) {
      sum_result = 1.0 / sum_result;
    }
    
    return sum_result;    
  }

};

#endif
