/**
 * @file test_functions.cpp
 * @author Ryan Curtin
 *
 * Implementations of the test functions defined in test_functions.hpp.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "test_functions.hpp"

using namespace mlpack::optimization::test;

//
// RosenbrockFunction implementation
//

RosenbrockFunction::RosenbrockFunction()
{
  initialPoint.set_size(2, 1);
  initialPoint[0] = -1.2;
  initialPoint[1] = 1;
}

/**
 * Calculate the objective function.
 */
double RosenbrockFunction::Evaluate(const arma::mat& coordinates)
{
  double x1 = coordinates[0];
  double x2 = coordinates[1];

  double objective = /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
                     /* f2(x) */ std::pow(1 - x1, 2);

  return objective;
}

/**
 * Calculate the gradient.
 */
void RosenbrockFunction::Gradient(const arma::mat& coordinates,
                                  arma::mat& gradient)
{
  // f'_{x1}(x) = -2 (1 - x1) + 400 (x1^3 - (x2 x1))
  // f'_{x2}(x) = 200 (x2 - x1^2)

  double x1 = coordinates[0];
  double x2 = coordinates[1];

  gradient.set_size(2, 1);
  gradient[0] = -2 * (1 - x1) + 400 * (std::pow(x1, 3) - x2 * x1);
  gradient[1] = 200 * (x2 - std::pow(x1, 2));
}

const arma::mat& RosenbrockFunction::GetInitialPoint() const
{
  return initialPoint;
}

//
// WoodFunction implementation
//

WoodFunction::WoodFunction()
{
  initialPoint.set_size(4, 1);
  initialPoint[0] = -3;
  initialPoint[1] = -1;
  initialPoint[2] = -3;
  initialPoint[3] = -1;
}

/**
 * Calculate the objective function.
 */
double WoodFunction::Evaluate(const arma::mat& coordinates)
{
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

/**
 * Calculate the gradient.
 */
void WoodFunction::Gradient(const arma::mat& coordinates,
                            arma::mat& gradient)
{
  // For convenience; we assume these temporaries will be optimized out.
  double x1 = coordinates[0];
  double x2 = coordinates[1];
  double x3 = coordinates[2];
  double x4 = coordinates[3];

  // f'_{x1}(x) = 400 (x1^3 - x2 x1) - 2 (1 - x1)
  // f'_{x2}(x) = 200 (x2 - x1^2) + 20 (x2 + x4 - 2) + (1 / 5) (x2 - x4)
  // f'_{x3}(x) = 360 (x3^3 - x4 x3) - 2 (1 - x3)
  // f'_{x4}(x) = 180 (x4 - x3^2) + 20 (x2 + x4 - 2) - (1 / 5) (x2 - x4)
  gradient.set_size(4, 1);
  gradient[0] = 400 * (std::pow(x1, 3) - x2 * x1) - 2 * (1 - x1);
  gradient[1] = 200 * (x2 - std::pow(x1, 2)) + 20 * (x2 + x4 - 2) +
      (1 / 5) * (x2 - x4);
  gradient[2] = 360 * (std::pow(x3, 3) - x4 * x3) - 2 * (1 - x3);
  gradient[3] = 180 * (x4 - std::pow(x3, 2)) + 20 * (x2 + x4 - 2) -
      (1 / 5) * (x2 - x4);
}

const arma::mat& WoodFunction::GetInitialPoint() const
{
  return initialPoint;
}

//
// GeneralizedRosenbrockFunction implementation
//

GeneralizedRosenbrockFunction::GeneralizedRosenbrockFunction(int n) : n(n)
{
  initialPoint.set_size(n, 1);
  for (int i = 0; i < n; i++) // Set to [-1.2 1 -1.2 1 ...].
  {
    if (i % 2 == 1)
      initialPoint[i] = -1.2;
    else
      initialPoint[i] = 1;
  }
}

/**
 * Calculate the objective function.
 */
double GeneralizedRosenbrockFunction::Evaluate(const arma::mat& coordinates)
    const
{
  double fval = 0;
  for (int i = 0; i < (n - 1); i++)
  {
    fval += 100 * std::pow(std::pow(coordinates[i], 2) -
        coordinates[i + 1], 2) + std::pow(1 - coordinates[i], 2);
  }

  return fval;
}

/**
 * Calculate the gradient.
 */
void GeneralizedRosenbrockFunction::Gradient(const arma::mat& coordinates,
                                             arma::mat& gradient) const
{
  gradient.set_size(n);
  for (int i = 0; i < (n - 1); i++)
  {
    gradient[i] = 400 * (std::pow(coordinates[i], 3) - coordinates[i] *
        coordinates[i + 1]) + 2 * (coordinates[i] - 1);

    if (i > 0)
      gradient[i] += 200 * (coordinates[i] - std::pow(coordinates[i - 1], 2));
  }

  gradient[n - 1] = 200 * (coordinates[n - 1] -
      std::pow(coordinates[n - 2], 2));
}

//! Calculate the objective function of one of the individual functions.
double GeneralizedRosenbrockFunction::Evaluate(const arma::mat& coordinates,
                                               const size_t i) const
{
  return 100 * std::pow((std::pow(coordinates[i], 2) - coordinates[i + 1]), 2) +
      std::pow(1 - coordinates[i], 2);
}

//! Calculate the gradient of one of the individual functions.
void GeneralizedRosenbrockFunction::Gradient(const arma::mat& coordinates,
                                             const size_t i,
                                             arma::mat& gradient) const
{
  gradient.zeros(n);

  gradient[i] = 400 * (std::pow(coordinates[i], 3) - coordinates[i] *
      coordinates[i + 1]) + 2 * (coordinates[i] - 1);
  gradient[i + 1] = 200 * (coordinates[i + 1] - std::pow(coordinates[i], 2));
}

const arma::mat& GeneralizedRosenbrockFunction::GetInitialPoint() const
{
  return initialPoint;
}

//
// RosenbrockWoodFunction implementation
//

RosenbrockWoodFunction::RosenbrockWoodFunction() : rf(4), wf()
{
  initialPoint.set_size(4, 2);
  initialPoint.col(0) = rf.GetInitialPoint();
  initialPoint.col(1) = wf.GetInitialPoint();
}

/**
 * Calculate the objective function.
 */
double RosenbrockWoodFunction::Evaluate(const arma::mat& coordinates)
{
  double objective = rf.Evaluate(coordinates.col(0)) +
                     wf.Evaluate(coordinates.col(1));

  return objective;
}

/***
 * Calculate the gradient.
 */
void RosenbrockWoodFunction::Gradient(const arma::mat& coordinates,
                                      arma::mat& gradient)
{
  gradient.set_size(4, 2);

  arma::vec grf(4);
  arma::vec gwf(4);

  rf.Gradient(coordinates.col(0), grf);
  wf.Gradient(coordinates.col(1), gwf);

  gradient.col(0) = grf;
  gradient.col(1) = gwf;
}

const arma::mat& RosenbrockWoodFunction::GetInitialPoint() const
{
  return initialPoint;
}
