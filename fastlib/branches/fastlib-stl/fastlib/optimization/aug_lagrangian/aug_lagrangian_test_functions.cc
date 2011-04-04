/**
 * @file aug_lagrangian_test_functions.cc
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangianTestFunction class.
 */

#include "aug_lagrangian_test_functions.h"

using namespace mlpack;
using namespace mlpack::optimization;

AugLagrangianTestFunction::AugLagrangianTestFunction() {
  // Set the initial point to be (0, 0).
  initial_point_.zeros(2, 1);
}

double AugLagrangianTestFunction::Evaluate(const arma::mat& coordinates) {
  // f(x) = 6 x_1^2 + 4 x_1 x_2 + 3 x_2^2
  return ((6 * std::pow(coordinates[0], 2)) +
          (4 * (coordinates[0] * coordinates[1])) +
          (3 * std::pow(coordinates[1], 2)));
}

void AugLagrangianTestFunction::Gradient(const arma::mat& coordinates,
                                         arma::mat& gradient) {
  // f'_x1(x) = 12 x_1 + 4 x_2
  // f'_x2(x) = 4 x_1 + 6 x_2
  gradient.set_size(2, 1);

  gradient[0] = 12 * coordinates[0] + 4 * coordinates[1];
  gradient[1] = 4 * coordinates[0] + 6 * coordinates[1];
}

double AugLagrangianTestFunction::EvaluateConstraint(int index,
    const arma::mat& coordinates) {
  // We return 0 if the index is wrong (not 0).
  if (index != 0)
    return 0;

  // c(x) = x_1 + x_2 - 5
  return (coordinates[0] + coordinates[1] - 5);
}

void AugLagrangianTestFunction::GradientConstraint(int index,
    const arma::mat& coordinates, arma::mat& gradient) {
  // If the user passed an invalid index (not 0), we will return a zero
  // gradient.
  gradient.zeros(2, 1);

  if (index == 0) {
    // c'_x1(x) = 1
    // c'_x2(x) = 1
    gradient.ones(2, 1); // Use a shortcut instead of assigning individually.
  }
}
