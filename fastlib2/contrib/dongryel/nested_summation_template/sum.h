#ifndef SUM_H
#define SUM_H

#include "fastlib/fastlib.h"
#include "operator.h"

class Sum: public Operator {

 public:

  double NaiveCompute() {

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

    return sum_result;
  }

  double MonteCarloCompute() {

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

    return sum_result;
  }

};

#endif
