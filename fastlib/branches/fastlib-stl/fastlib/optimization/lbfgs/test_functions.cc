/***
 * test_functions.cc
 *
 * Implementations of the test functions defined in test_functions.h.
 *
 * @author Ryan Curtin
 */
#include "test_functions.h"

using namespace mlpack::optimization::test;

RosenbrockFunction::RosenbrockFunction() {
  initial_point.set_size(2);
  initial_point[0] = -1.2;
  initial_point[1] = 1;
}

/***
 * Calculate the gradient.
 */
void RosenbrockFunction::Gradient(const arma::vec& coordinates,
                                  arma::vec& gradient) {
  // f1(x) = 100 (x2 - x1^2)^2
  // f2(x) = (1 - x1)^2

  // f1'(x) = 200 (x2 - x1^2) * (1 - 2 * x1)
  // f2'(x) = -2 * (1 - x1)

  double x1 = coordinates[0];
  double x2 = coordinates[1];
 
  gradient.set_size(2);
  gradient[0] = -2 * (1 - x1) + 400 * (std::pow(x1, 3) - x2 * x1);
  gradient[1] = 200 * (x2 - std::pow(x1, 2));
}

/***
 * Calculate the objective function.
 */
double RosenbrockFunction::Evaluate(const arma::vec& coordinates) {
  double x1 = coordinates[0];
  double x2 = coordinates[1];

  double objective = /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2.0), 2.0) +
                     /* f2(x) */ std::pow(1 - x1, 2.0);

  return objective;
}

const arma::vec& RosenbrockFunction::GetInitialPoint() {
  return initial_point;
}
