/***
 * @file aug_lagrangian_test.h
 *
 * Define a test function for the augmented Lagrangian method.
 */

#ifndef __AUG_LAGRANGIAN_TEST_FUNCTIONS_H
#define __AUG_LAGRANGIAN_TEST_FUNCTIONS_H

#include <fastlib/fastlib.h>
#include <armadillo>

namespace mlpack {
namespace optimization {

/***
 * This function is taken from "Practical Mathematical Optimization" (Snyman),
 * section 5.3.8 ("Application of the Augmented Lagrangian Method").  It has
 * only one constraint.
 *
 * The minimum that satisfies the constraint is x = [1, 4], with an objective
 * value of 70.
 */
class AugLagrangianTestFunction {
 public:
  AugLagrangianTestFunction();

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  int NumConstraints() const { return 1; }

  double EvaluateConstraint(int index, const arma::mat& coordinates);
  void GradientConstraint(int index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint() { return initial_point_; }

 private:
  arma::mat initial_point_;
};

/***
 * This function is taken from M. Gockenbach's lectures on general nonlinear
 * programs, found at:
 * http://www.math.mtu.edu/~msgocken/ma5630spring2003/lectures/nlp/nlp.pdf
 *
 * The program we are using is example 2.5 from this document.
 * I have arbitrarily decided that this will be called the Gockenbach function.
 *
 * The minimum that satisfies the two constraints is given as
 *   x = [0.12288, -1.1078, 0.015100], with an objective value of about 29.634.
 */
class GockenbachFunction {
 public:
  GockenbachFunction();

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  int NumConstraints() const { return 2; };

  double EvaluateConstraint(int index, const arma::mat& coordinates);
  void GradientConstraint(int index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint() { return initial_point_; }

 private:
  arma::mat initial_point_;
};

}; // namespace optimization
}; // namespace mlpack

#endif
