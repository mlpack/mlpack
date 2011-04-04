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

}; // namespace optimization
}; // namespace mlpack

#endif
