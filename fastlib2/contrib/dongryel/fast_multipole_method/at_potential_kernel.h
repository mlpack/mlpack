#ifndef AT_POTENTIAL_KERNEL_H
#define AT_POTENTIAL_KERNEL_H

class ATPotentialKernel {

 public:

  static const int order = 3;

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
  
  double PositiveEvaluate(double sq_dist1, double sq_dist2, double sq_dist3) {
    
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

    return (first_part + second_part + third_part) /
      (8.0 * fifth_dist1 * fifth_dist2 * fifth_dist3);
  }

  double NegativeEvaluate(double sq_dist1, double sq_dist2, double sq_dist3) {

    double sixth_dist1 = sq_dist1 * sq_dist1 * sq_dist1;
    double sixth_dist2 = sq_dist2 * sq_dist2 * sq_dist2;
    double sixth_dist3 = sq_dist3 * sq_dist3 * sq_dist3;
    double fifth_dist1 = sq_dist1 * sq_dist1 * sqrt(sq_dist1);
    double fifth_dist2 = sq_dist2 * sq_dist2 * sqrt(sq_dist2);
    double fifth_dist3 = sq_dist3 * sq_dist3 * sqrt(sq_dist3);

    return -0.375 * (sixth_dist1 + sixth_dist2 + sixth_dist3) /
      (fifth_dist1 * fifth_dist2 * fifth_dist3);
  }
};

#endif
