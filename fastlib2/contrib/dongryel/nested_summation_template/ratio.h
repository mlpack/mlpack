#ifndef RATIO_H
#define RATIO_H

#include "operator.h"

class Ratio: public Operator {

 public:
  
  double NaiveCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double numerator_result = operators_[0]->NaiveCompute();
    double denominator_result = operators_[1]->NaiveCompute();    
    double result = numerator_result / denominator_result;

    if(!is_positive_) {
      result = -result;
    }
    if(should_be_inverted_) {
      result = 1.0 / result;
    }
    
    return result;
  }

  double MonteCarloCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double numerator_result = operators_[0]->MonteCarloCompute();
    double denominator_result = operators_[1]->MonteCarloCompute();    
    double result = numerator_result / denominator_result;

    if(!is_positive_) {
      result = -result;
    }
    if(should_be_inverted_) {
      result = 1.0 / result;
    }
    
    return result;
  }

};

#endif
