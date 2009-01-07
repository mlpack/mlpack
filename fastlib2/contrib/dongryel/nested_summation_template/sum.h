#ifndef SUM_H
#define SUM_H

#include "fastlib/fastlib.h"
#include "operator.h"

template<is_positive>
class Sum: public Operator {

 public:

  double NaiveCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double sum_result = 0;

    // If this is the base case, where there is no more nested
    // operator inside, then call the multi-tree algorithm.
    if(indices_.size() > 0) {
      
    }

    // Otherwise, recursively evaluate the operators and add them up.
    else {
      
      for(index_t i = 0; i < operators_.size(); i++) {
	sum_result += operators_[i]->NaiveCompute();
      }
    }

    if(is_positive) {
      return sum_result;
    }
    else {
      return -sum_result;
    }
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

    if(is_positive) {
      return sum_result;
    }
    else {
      return -sum_result;
    }
  }

};

#endif
