#ifndef RATIO_H
#define RATIO_H

#include "operator.h"

class Ratio: public Operator {

 public:
  
  double NaiveCompute
  (std::map<index_t, index_t> &constant_dataset_indices) {

    double numerator_result = 
      operators_[0]->NaiveCompute(constant_dataset_indices);
    double denominator_result = 
      operators_[1]->NaiveCompute(constant_dataset_indices);
    double result = numerator_result / denominator_result;
    
    return PostProcess_(constant_dataset_indices, result);
  }

  double MonteCarloCompute
  (std::map<index_t, index_t> &constant_dataset_indices) {

    double numerator_result = 
      operators_[0]->MonteCarloCompute(constant_dataset_indices);
    double denominator_result = 
      operators_[1]->MonteCarloCompute(constant_dataset_indices);
    double result = numerator_result / denominator_result;

    return PostProcess_(constant_dataset_indices, result);
  }

};

#endif
