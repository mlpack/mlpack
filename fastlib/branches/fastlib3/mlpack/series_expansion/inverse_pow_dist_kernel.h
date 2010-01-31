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
  
  double EvalUnnorm(const double *point1, const double *point2,
		    double squared_distance) const {
    return (point1[dimension_] - point2[dimension_]) / 
      pow(squared_distance, lambda_ / 2.0);
  }

  static inline double EvalUnnorm(int dimension_in, double lambda_in,
				  const double *point1, const double *point2,
				  double squared_distance) {
    return (point1[dimension_in] - point2[dimension_in]) / 
      pow(squared_distance, lambda_in / 2.0);
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

    if(lambda_ > 0) {
      return 1.0 / pow(sqdist, lambda_ / 2.0);
    }
    else {
      return pow(sqdist, -lambda_ / 2.0);
    }
  }

  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }

  double EvalUnnormOnSq(double sqdist) const {
    if(lambda_ > 0) {
      return 1.0 / pow(sqdist, lambda_ / 2.0);
    }
    else {
      return pow(sqdist, -lambda_ / 2.0);
    }
  }

  static inline double EvalUnnormOnSq(double lambda_in, double sqdist) {
    if(lambda_in > 0) {
      return 1.0 / pow(sqdist, lambda_in / 2.0);
    }
    else {
      return pow(sqdist, -lambda_in / 2.0);
    }
  }

  DRange RangeUnnormOnSq(const DRange &range) const {
    return DRange(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
  }

};

#endif
