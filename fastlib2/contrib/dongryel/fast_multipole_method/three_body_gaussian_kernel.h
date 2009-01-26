#ifndef THREE_BODY_GAUSSIAN_KERNEL_H
#define THREE_BODY_GAUSSIAN_KERNEL_H

#include "mbp_kernel.h"

class ThreeBodyGaussianKernel: public MultibodyPotentialKernel {

 private:
  double bandwidth_sq_;
  double inv_bandwidth_sq_;

 public:

  static const int order = 3;

  void Init(double bandwidth_in) {
    bandwidth_sq_ = math::Sqr(bandwidth_in);
    inv_bandwidth_sq_ = 1.0 / bandwidth_sq_;
  }

  void PositiveEvaluate(const ArrayList<index_t> &indices, 
			const ArrayList<Matrix *> &sets,
			ArrayList<DRange> &positive_potential_bounds) {
    
    double sum_squared_distances = min_squared_distances.get(0, 1) +
      min_squared_distances.get(0, 2) + min_squared_distances.get(1, 2);
    double positive_potential =
      exp(-0.5 * inv_bandwidth_sq_ * sum_squared_distances);

    positive_potential_bounds[indices[0]] += positive_potential;
    positive_potential_bounds[indices[1]] += positive_potential;
    positive_potential_bounds[indices[2]] += positive_potential;
  }

  void NegativeEvaluate(const ArrayList<index_t> &indices, 
			const ArrayList<Matrix *> &sets,
			ArrayList<DRange> &negative_potential_bounds) {

    // Do nothing since there is no negative contribution...
  }

};

#endif
