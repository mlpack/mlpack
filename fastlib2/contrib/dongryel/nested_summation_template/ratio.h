#ifndef RATIO_H
#define RATIO_H

#include "operator.h"

template<is_ratio>
class Ratio: public Operator {
  
 public:
  
  double NaiveCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double numerator_result = operators_[0]->NaiveCompute();
    double denominator_result = operators_[1]->NaiveCompute();
    
    if(is_ratio_) {
      return numerator_result / denominator_result;
    }
    else {
      return numerator_result * denominator_result;
    }
  }

  double MonteCarloCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {

    double numerator_result = operators_[0]->MonteCarloCompute();
    double denominator_result = operators_[1]->MonteCarloCompute();

    if(is_ratio_) {
      return numerator_result / denominator_result;
    }
    else {
      return numerator_result * denominator_result;
    }
  }

};

#endif
