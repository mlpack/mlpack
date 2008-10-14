#ifndef MONOMIAL_KERNEL_H
#define MONOMIAL_KERNEL_H

#include "fastlib/fastlib.h"

class MonomialKernel {
  
 public:
  double power_;
  index_t dimension_;
  
  OT_DEF_BASIC(MonomialKernel) {
    OT_MY_OBJECT(power_);
    OT_MY_OBJECT(dimension_);
  }

 public:
  
  void Init(double power_in, index_t dimension_in) {
    power_ = power_in;
    dimension_ = dimension_in;
  }
  
  double EvalUnnorm(const double *point) const {
    double sqdist = la::Dot(dimension_, point, point);
    return pow(sqdist, power_ / 2.0);
  }

};

#endif
