#ifndef INVERSE_POW_DIST_KERNEL_H
#define INVERSE_POW_DIST_KERNEL_H

#include "fastlib/fastlib.h"

class InversePowDistKernel {
  
 private:
  double lambda_;
  
 public:

  void Init(double lambda_in) {
    lambda_ = lambda_in;
  }

  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }

  double EvalUnnormOnSq(double sqdist) const {
    return 1.0 / pow(sqdist, lambda_ / 2.0);
  }

  DRange RangeUnnormOnSq(const DRange &range) const {
    return DRange(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
  }

};

#endif
