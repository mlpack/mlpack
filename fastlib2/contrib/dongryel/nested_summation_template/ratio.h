#ifndef RATIO_H
#define RATIO_H

#include "operator.h"

class Ratio: public Operator {

 public:
  
  double NaiveCompute() {

    double numerator_result = operators_[0]->NaiveCompute();
    double denominator_result = operators_[1]->NaiveCompute();

    return numerator_result / denominator_result;
  }

  double MonteCarloCompute() {

    double numerator_result = operators_[0]->MonteCarloCompute();
    double denominator_result = operators_[1]->MonteCarloCompute();

    return numerator_result / denominator_result;
  }

};

#endif
