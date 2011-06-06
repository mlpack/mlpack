/**
 * @file aug_lagrangian_test_functions.cc
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangianTestFunction class.
 */

#include "aug_lagrangian_test_functions.h"

using namespace mlpack;
using namespace mlpack::optimization;

//
// AugLagrangianTestFunction
//
AugLagrangianTestFunction::AugLagrangianTestFunction() {
  // Set the initial point to be (0, 0).
  initial_point_.zeros(2, 1);
}

AugLagrangianTestFunction::AugLagrangianTestFunction(
      const arma::mat& initial_point) :
    initial_point_(initial_point) {
  // Nothing to do.
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

// 
// GockenbachFunction
//
GockenbachFunction::GockenbachFunction() {
  // Set the initial point to (0, 0, 1).
  initial_point_.zeros(3, 1);
  initial_point_[2] = 1;
}

GockenbachFunction::GockenbachFunction(const arma::mat& initial_point) :
    initial_point_(initial_point) {
  // Nothing to do.
}

double GockenbachFunction::Evaluate(const arma::mat& coordinates) {
  // f(x) = (x_1 - 1)^2 + 2 (x_2 + 2)^2 + 3(x_3 + 3)^2
  return ((std::pow(coordinates[0] - 1, 2)) +
          (2 * std::pow(coordinates[1] + 2, 2)) +
          (3 * std::pow(coordinates[2] + 3, 2)));
}

void GockenbachFunction::Gradient(const arma::mat& coordinates,
                                  arma::mat& gradient) {
  // f'_x1(x) = 2 (x_1 - 1)
  // f'_x2(x) = 4 (x_2 + 2)
  // f'_x3(x) = 6 (x_3 + 3)
  gradient.set_size(3, 1);

  gradient[0] = 2 * (coordinates[0] - 1);
  gradient[1] = 4 * (coordinates[1] + 2);
  gradient[2] = 6 * (coordinates[2] + 3);
}

double GockenbachFunction::EvaluateConstraint(int index,
                                              const arma::mat& coordinates) {
  double constraint = 0;
  
  switch(index) {
    case 0: // g(x) = (x_3 - x_2 - x_1 - 1) = 0
      constraint = (coordinates[2] - coordinates[1] - coordinates[0] - 1);
      break;

    case 1: // h(x) = (x_3 - x_1^2) >= 0
      // To deal with the inequality, the constraint will simply evaluate to 0
      // when h(x) >= 0.
      constraint = std::min(0.0,
          (coordinates[2] - std::pow(coordinates[0], 2)));
      break;
  }

  // 0 will be returned for an invalid index (but this is okay).
  return constraint;
}

void GockenbachFunction::GradientConstraint(int index,
                                            const arma::mat& coordinates,
                                            arma::mat& gradient) {
  gradient.zeros(3, 1);

  switch(index) {
    case 0:
      // g'_x1(x) = -1
      // g'_x2(x) = -1
      // g'_x3(x) = 1
      gradient[0] = -1;
      gradient[1] = -1;
      gradient[2] = 1;
      break;

    case 1:
      // h'_x1(x) = -2 x_1
      // h'_x2(x) = 0
      // h'_x3(x) = 1
      gradient[0] = -2 * coordinates[0];
      gradient[2] = 1;
      break;
  }
}
