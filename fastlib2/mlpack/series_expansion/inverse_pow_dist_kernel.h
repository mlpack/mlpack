#ifndef INVERSE_POW_DIST_KERNEL_H
#define INVERSE_POW_DIST_KERNEL_H

#include "fastlib/fastlib.h"

class InversePowDistGradientKernel {

 public:
  double lambda_;
  
  index_t dimension_;

 public:
  
  void Init(double lambda_in, index_t dimension_in) {
    lambda_ = lambda_in;
    dimension_ = dimension_in;
  }

  double EvalUnnorm(const double *point) const {
    double sqdist = la::Dot(dimension_, point, point);
    return point[dimension_] / pow(sqdist, lambda_ / 2.0);
  }
};

class InversePowDistKernel {
  
 public:
  double lambda_;
  
  index_t dimension_;

 public:

  void Init(double lambda_in, index_t dimension_in) {
    lambda_ = lambda_in;
    dimension_ = dimension_in;
  }

  double EvalUnnorm(const double *point) const {
    double sqdist = la::Dot(dimension_, point, point);
    return 1.0 / pow(sqdist, lambda_ / 2.0);
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
