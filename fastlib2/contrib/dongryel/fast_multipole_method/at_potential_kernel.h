#ifndef AT_POTENTIAL_KERNEL_H
#define AT_POTENTIAL_KERNEL_H

#include "mbp_kernel.h"

class ATPotentialKernel: public MultibodyPotentialKernel {

 private:

  double PositiveEvaluateCommon_(const Matrix &squared_distances) {

    double sq_dist1 = squared_distances.get(0, 1);
    double sq_dist2 = squared_distances.get(0, 2);
    double sq_dist3 = squared_distances.get(1, 2);

    double fourth_dist1 = sq_dist1 * sq_dist1;
    double fourth_dist2 = sq_dist2 * sq_dist2;
    double fourth_dist3 = sq_dist3 * sq_dist3;
    double fifth_dist1 = fourth_dist1 * sqrt(sq_dist1);
    double fifth_dist2 = fourth_dist2 * sqrt(sq_dist2);
    double fifth_dist3 = fourth_dist3 * sqrt(sq_dist3);
    double first_part = 3.0 * fourth_dist1 * (sq_dist2 + sq_dist3);
    double second_part = 3.0 * sq_dist2 * sq_dist3 * (sq_dist2 + sq_dist3);
    double third_part = sq_dist1 * 
      (3.0 * fourth_dist2 + 2.0 * sq_dist2 * sq_dist3 + 3.0 * fourth_dist3);
    double common_contribution = (first_part + second_part + third_part) /
      (8.0 * fifth_dist1 * fifth_dist2 * fifth_dist3);

    return common_contribution;
  }

  double NegativeEvaluateCommon_(const Matrix &squared_distances) {

    double sq_dist1 = squared_distances.get(0, 1);
    double sq_dist2 = squared_distances.get(0, 2);
    double sq_dist3 = squared_distances.get(1, 2);

    double sixth_dist1 = sq_dist1 * sq_dist1 * sq_dist1;
    double sixth_dist2 = sq_dist2 * sq_dist2 * sq_dist2;
    double sixth_dist3 = sq_dist3 * sq_dist3 * sq_dist3;
    double fifth_dist1 = sq_dist1 * sq_dist1 * sqrt(sq_dist1);
    double fifth_dist2 = sq_dist2 * sq_dist2 * sqrt(sq_dist2);
    double fifth_dist3 = sq_dist3 * sq_dist3 * sqrt(sq_dist3);
    
    double common_contribution = 
      -0.375 * (sixth_dist1 + sixth_dist2 + sixth_dist3) /
      (fifth_dist1 * fifth_dist2 * fifth_dist3);    
    return common_contribution;
  }

 public:

  static const int order = 3;

  void Init(double bandwidth_in) {

    // Ignore the bandwidth parameter, but initialize the min and max
    // distance scratch matrices...
    min_squared_distances.Init(3, 3);
    max_squared_distances.Init(3, 3);

    // Initialize the zeros.
    SetZero();
  }

  double Gradient(double distance) {
    
    // COMPLETE ME!!!
    return 0;
  }

  double Evaluate(double sq_dist1, double sq_dist2, double sq_dist3) {

    double cube_dist1 = sq_dist1 * sqrt(sq_dist1);
    double cube_dist2 = sq_dist2 * sqrt(sq_dist2);
    double cube_dist3 = sq_dist3 * sqrt(sq_dist3);

    double part1 = 1 / (cube_dist1 * cube_dist2 * cube_dist3);
    double part2 = 0.375 * (sq_dist1 + sq_dist2 - sq_dist3) *
      (sq_dist1 - sq_dist2 + sq_dist3) * (-sq_dist1 + sq_dist2 + sq_dist3) /
      (cube_dist1 * sq_dist1 * cube_dist2 * sq_dist2 * cube_dist3 *
       sq_dist3);

    return part1 + part2;
  }

  double EvalUnnorm(double distance) {

    return 1.0 / (DBL_EPSILON + pow(distance, 2.0));
  }

};

#endif
