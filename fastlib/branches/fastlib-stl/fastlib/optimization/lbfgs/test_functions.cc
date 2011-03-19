/***
 * test_functions.cc
 *
 * Implementations of the test functions defined in test_functions.h.
 *
 * @author Ryan Curtin
 */
#include "test_functions.h"

using namespace mlpack::optimization::test;

//
// RosenbrockFunction implementation
//

RosenbrockFunction::RosenbrockFunction() {
  initial_point.set_size(2);
  initial_point[0] = -1.2;
  initial_point[1] = 1;
}

/***
 * Calculate the objective function.
 */
double RosenbrockFunction::Evaluate(const arma::vec& coordinates) {
  double x1 = coordinates[0];
  double x2 = coordinates[1];

  double objective = /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
                     /* f2(x) */ std::pow(1 - x1, 2);

  return objective;
}

/***
 * Calculate the gradient.
 */
void RosenbrockFunction::Gradient(const arma::vec& coordinates,
                                  arma::vec& gradient) {
  // f'_{x1}(x) = -2 (1 - x1) + 400 (x1^3 - (x2 x1))
  // f'_{x2}(x) = 200 (x2 - x1^2)

  double x1 = coordinates[0];
  double x2 = coordinates[1];
 
  gradient.set_size(2);
  gradient[0] = -2 * (1 - x1) + 400 * (std::pow(x1, 3) - x2 * x1);
  gradient[1] = 200 * (x2 - std::pow(x1, 2));
}

const arma::vec& RosenbrockFunction::GetInitialPoint() {
  return initial_point;
}

//
// WoodFunction implementation
//

WoodFunction::WoodFunction() {
  initial_point.set_size(4);
  initial_point[0] = -3;
  initial_point[1] = -1;
  initial_point[2] = -3;
  initial_point[3] = -1;
}

/***
 * Calculate the objective function.
 */
double WoodFunction::Evaluate(const arma::vec& coordinates) {
  // For convenience; we assume these temporaries will be optimized out.
  double x1 = coordinates[0];
  double x2 = coordinates[1];
  double x3 = coordinates[2];
  double x4 = coordinates[3];

  double objective = /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
                     /* f2(x) */ std::pow(1 - x1, 2) +
                     /* f3(x) */ 90 * std::pow(x4 - std::pow(x3, 2), 2) +
                     /* f4(x) */ std::pow(1 - x3, 2) +
                     /* f5(x) */ 10 * std::pow(x2 + x4 - 2, 2) +
                     /* f6(x) */ (1 / 10) * std::pow(x2 - x4, 2);

  return objective;
}

/***
 * Calculate the gradient.
 */
void WoodFunction::Gradient(const arma::vec& coordinates,
                            arma::vec& gradient) {
  // For convenience; we assume these temporaries will be optimized out.
  double x1 = coordinates[0];
  double x2 = coordinates[1];
  double x3 = coordinates[2];
  double x4 = coordinates[3];

  // f'_{x1}(x) = 400 (x1^3 - x2 x1) - 2 (1 - x1)
  // f'_{x2}(x) = 200 (x2 - x1^2) + 20 (x2 + x4 - 2) + (1 / 5) (x2 - x4)
  // f'_{x3}(x) = 360 (x3^3 - x4 x3) - 2 (1 - x3)
  // f'_{x4}(x) = 180 (x4 - x3^2) + 20 (x2 + x4 - 2) - (1 / 5) (x2 - x4)
  gradient.set_size(4);
  gradient[0] = 400 * (std::pow(x1, 3) - x2 * x1) - 2 * (1 - x1);
  gradient[1] = 200 * (x2 - std::pow(x1, 2)) + 20 * (x2 + x4 - 2) +
      (1 / 5) * (x2 - x4);
  gradient[2] = 360 * (std::pow(x3, 3) - x4 * x3) - 2 * (1 - x3);
  gradient[3] = 180 * (x4 - std::pow(x3, 2)) + 20 * (x2 + x4 - 2) -
      (1 / 5) * (x2 - x4);
}

const arma::vec& WoodFunction::GetInitialPoint() {
  return initial_point;
}

//
// GeneralizedRosenbrockFunction implementation
//

GeneralizedRosenbrockFunction::GeneralizedRosenbrockFunction(int n) : n(n) {
  initial_point.set_size(n);
  for (int i = 0; i < n; i++) { // Set to [-1.2 1 -1.2 1 ...].
    if (i % 2 == 1)
      initial_point[i] = -1.2;
    else
      initial_point[i] = 1;
  }
}

/***
 * Calculate the objective function.
 */
double GeneralizedRosenbrockFunction::Evaluate(const arma::vec& coordinates) {
  double fval = 0;
  for (int i = 0; i < (n - 1); i++) {
    fval += 100 * std::pow(std::pow(coordinates[i], 2) -
        coordinates[i + 1], 2) + std::pow(1 - coordinates[i], 2);
  }

  return fval;
}

/***
 * Calculate the gradient.
 */
void GeneralizedRosenbrockFunction::Gradient(const arma::vec& coordinates,
                                             arma::vec& gradient) {
  gradient.set_size(n);
  for(int i = 0; i < (n - 1); i++) {
    gradient[i] = 400 * (std::pow(coordinates[i], 3) - coordinates[i] * 
        coordinates[i + 1]) + 2 * (coordinates[i] - 1);
    if(i > 0)
      gradient[i] += 200 * (coordinates[i] - std::pow(coordinates[i - 1], 2));
  }

  gradient[n - 1] = 200 * (coordinates[n - 1] -
      std::pow(coordinates[n - 2], 2)); 
}

const arma::vec& GeneralizedRosenbrockFunction::GetInitialPoint() {
  return initial_point;
}
