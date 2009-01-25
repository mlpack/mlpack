#ifndef THREE_BODY_GAUSSIAN_KERNEL_H
#define THREE_BODY_GAUSSIAN_KERNEL_H

class ThreeBodyGaussianKernel {

 private:
  double bandwidth_sq_;
  double inv_bandwidth_sq_;

 public:

  static const int order = 3;

  void Init(double bandwidth_in) {
    bandwidth_sq_ = math::Sqr(bandwidth_in);
    inv_bandwidth_sq_ = 1.0 / bandwidth_sq_;
  }

  double Evaluate(double sq_dist1, double sq_dist2, double sq_dist3) {
    
    return PositiveEvaluate(sq_dist1, sq_dist2, sq_dist3);
  }
  
  double PositiveEvaluate(double sq_dist1, double sq_dist2, double sq_dist3) {
    return exp(-0.5 * (sq_dist1 + sq_dist2 + sq_dist3) * inv_bandwidth_sq_);
  }

};

#endif
