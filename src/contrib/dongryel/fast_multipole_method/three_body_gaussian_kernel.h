#ifndef THREE_BODY_GAUSSIAN_KERNEL_H
#define THREE_BODY_GAUSSIAN_KERNEL_H

#include "mbp_kernel.h"

class ThreeBodyGaussianKernel: public MultibodyPotentialKernel {

 private:
  double bandwidth_sq_;
  double inv_bandwidth_sq_;

  double PositiveEvaluateCommon_(const Matrix &squared_distances) {

    double sum_squared_distances = squared_distances.get(0, 1) +
      squared_distances.get(0, 2) + squared_distances.get(1, 2);
    double positive_potential =
      exp(-0.5 * inv_bandwidth_sq_ * sum_squared_distances);

    return positive_potential;
  }

  double NegativeEvaluateCommon_(const Matrix &squared_distances) {
    return 0;
  }

 public:

  static const int order = 3;

  double EvalUnnorm(double distance) {

    return exp(-0.5 * math::Sqr(distance) * inv_bandwidth_sq_);
  }

  double Gradient(double distance) {
    
    return -distance * inv_bandwidth_sq_ * 
      exp(-0.5 * math::Sqr(distance) * inv_bandwidth_sq_);
  }

  void Init(double bandwidth_in) {
    bandwidth_sq_ = math::Sqr(bandwidth_in);
    inv_bandwidth_sq_ = 1.0 / bandwidth_sq_;

    min_squared_distances.Init(3, 3);
    max_squared_distances.Init(3, 3);

    SetZero();
  }

};

#endif
