/**
 * @file aug_lagrangian_test_functions.cpp
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangianTestFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "aug_lagrangian_test_functions.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

//
// AugLagrangianTestFunction
//
AugLagrangianTestFunction::AugLagrangianTestFunction()
{
  // Set the initial point to be (0, 0).
  initialPoint.zeros(2, 1);
}

AugLagrangianTestFunction::AugLagrangianTestFunction(
      const arma::mat& initialPoint) :
    initialPoint(initialPoint)
{
  // Nothing to do.
}

double AugLagrangianTestFunction::Evaluate(const arma::mat& coordinates)
{
  // f(x) = 6 x_1^2 + 4 x_1 x_2 + 3 x_2^2
  return ((6 * std::pow(coordinates[0], 2)) +
          (4 * (coordinates[0] * coordinates[1])) +
          (3 * std::pow(coordinates[1], 2)));
}

void AugLagrangianTestFunction::Gradient(const arma::mat& coordinates,
                                         arma::mat& gradient)
{
  // f'_x1(x) = 12 x_1 + 4 x_2
  // f'_x2(x) = 4 x_1 + 6 x_2
  gradient.set_size(2, 1);

  gradient[0] = 12 * coordinates[0] + 4 * coordinates[1];
  gradient[1] = 4 * coordinates[0] + 6 * coordinates[1];
}

double AugLagrangianTestFunction::EvaluateConstraint(const size_t index,
    const arma::mat& coordinates)
{
  // We return 0 if the index is wrong (not 0).
  if (index != 0)
    return 0;

  // c(x) = x_1 + x_2 - 5
  return (coordinates[0] + coordinates[1] - 5);
}

void AugLagrangianTestFunction::GradientConstraint(const size_t index,
    const arma::mat& /* coordinates */,
    arma::mat& gradient)
{
  // If the user passed an invalid index (not 0), we will return a zero
  // gradient.
  gradient.zeros(2, 1);

  if (index == 0)
  {
    // c'_x1(x) = 1
    // c'_x2(x) = 1
    gradient.ones(2, 1); // Use a shortcut instead of assigning individually.
  }
}

//
// GockenbachFunction
//
GockenbachFunction::GockenbachFunction()
{
  // Set the initial point to (0, 0, 1).
  initialPoint.zeros(3, 1);
  initialPoint[2] = 1;
}

GockenbachFunction::GockenbachFunction(const arma::mat& initialPoint) :
    initialPoint(initialPoint)
{
  // Nothing to do.
}

double GockenbachFunction::Evaluate(const arma::mat& coordinates)
{
  // f(x) = (x_1 - 1)^2 + 2 (x_2 + 2)^2 + 3(x_3 + 3)^2
  return ((std::pow(coordinates[0] - 1, 2)) +
          (2 * std::pow(coordinates[1] + 2, 2)) +
          (3 * std::pow(coordinates[2] + 3, 2)));
}

void GockenbachFunction::Gradient(const arma::mat& coordinates,
                                  arma::mat& gradient)
{
  // f'_x1(x) = 2 (x_1 - 1)
  // f'_x2(x) = 4 (x_2 + 2)
  // f'_x3(x) = 6 (x_3 + 3)
  gradient.set_size(3, 1);

  gradient[0] = 2 * (coordinates[0] - 1);
  gradient[1] = 4 * (coordinates[1] + 2);
  gradient[2] = 6 * (coordinates[2] + 3);
}

double GockenbachFunction::EvaluateConstraint(const size_t index,
                                              const arma::mat& coordinates)
{
  double constraint = 0;

  switch (index)
  {
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

void GockenbachFunction::GradientConstraint(const size_t index,
                                            const arma::mat& coordinates,
                                            arma::mat& gradient)
{
  gradient.zeros(3, 1);

  switch (index)
  {
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

//
// LovaszThetaSDP
//
LovaszThetaSDP::LovaszThetaSDP() : edges(0), vertices(0), initialPoint(0, 0)
{ }

LovaszThetaSDP::LovaszThetaSDP(const arma::mat& edges) : edges(edges),
    initialPoint(0, 0)
{
  // Calculate V by finding the maximum index in the edges matrix.
  vertices = max(max(edges)) + 1;
//  Log::Debug << vertices << " vertices in graph." << std::endl;
}

double LovaszThetaSDP::Evaluate(const arma::mat& coordinates)
{
  // The objective is equal to -Tr(ones * X) = -Tr(ones * (R^T * R)).
  // This can be simplified into the negative sum of (R^T * R).
//  Log::Debug << "Evaluting objective function with coordinates:" << std::endl;
//  std::cout << coordinates << std::endl;
//  Log::Debug << "trans(coord) * coord:" << std::endl;
//  std::cout << (trans(coordinates) * coordinates) << std::endl;


  arma::mat x = trans(coordinates) * coordinates;
  double obj = -accu(x);

//  double obj = 0;
//  for (size_t i = 0; i < coordinates.n_cols; i++)
//    obj -= dot(coordinates.col(i), coordinates.col(i));

//  Log::Debug << "Objective function is " << obj << "." << std::endl;

  return obj;
}

void LovaszThetaSDP::Gradient(const arma::mat& coordinates,
                              arma::mat& gradient)
{

  // The gradient is equal to (2 S' R^T)^T, with R being coordinates.
  // S' = C - sum_{i = 1}^{m} [ y_i - sigma (Tr(A_i * (R^T R)) - b_i)] * A_i
  // We will calculate it in a not very smart way, but it should work.

  // Initialize S' piece by piece.  It is of size n x n.
  const size_t n = coordinates.n_cols;
  arma::mat s(n, n);
  s.ones();
  s *= -1; // C = -ones().

  for (size_t i = 0; i < NumConstraints(); ++i)
  {
    // Calculate [ y_i - sigma (Tr(A_i * (R^T R)) - b_i) ] * A_i.
    // Result will be a matrix; inner result is a scalar.
    if (i == 0)
    {
      // A_0 = I_n.  Hooray!  That's easy!  b_0 = 1.
      double inner = -1 * double(n) - 0.5 *
          (accu(trans(coordinates) % coordinates) - 1);

      arma::mat zz = (inner * arma::eye<arma::mat>(n, n));

//      Log::Debug << "Constraint " << i << " matrix to add is " << std::endl;
//      Log::Debug << zz << std::endl;

      s -= zz;
    }
    else
    {
      // Get edge so we can construct constraint A_i matrix.  b_i = 0.
      arma::vec edge = edges.col(i - 1);

      arma::mat a;
      a.zeros(n, n);

      // Only two nonzero entries.
      a(edge[0], edge[1]) = 1;
      a(edge[1], edge[0]) = 1;

      double inner = (-1) - 0.5 *
          (accu(a % (trans(coordinates) * coordinates)));

      arma::mat zz = (inner * a);

//      Log::Debug << "Constraint " << i << " matrix to add is " << std::endl;
//      Log::Debug << zz << std::endl;

      s -= zz;
    }
  }

  gradient = trans(2 * s * trans(coordinates));

  // The gradient of -Tr(ones * X) is equal to -2 * ones * R
//  arma::mat ones;
//  ones.ones(coordinates.n_rows, coordinates.n_rows);
//  gradient = -2 * ones * coordinates;

//  Log::Debug << "Done with gradient." << std::endl;
//  std::cout << gradient;
}

size_t LovaszThetaSDP::NumConstraints() const
{
  // Each edge is a constraint, and we have the constraint Tr(X) = 1.
  return edges.n_cols + 1;
}

double LovaszThetaSDP::EvaluateConstraint(const size_t index,
                                          const arma::mat& coordinates)
{
  if (index == 0) // This is the constraint Tr(X) = 1.
  {
    double sum = -1; // Tr(X) - 1 = 0, so we prefix the subtraction.
    for (size_t i = 0; i < coordinates.n_cols; i++)
      sum += std::abs(dot(coordinates.col(i), coordinates.col(i)));

//    Log::Debug << "Constraint " << index << " evaluates to " << sum << std::endl;
    return sum;
  }

  size_t i = edges(0, index - 1);
  size_t j = edges(1, index - 1);

//  Log::Debug << "Constraint " << index << " evaluates to " <<
//    dot(coordinates.col(i), coordinates.col(j)) << "." << std::endl;

  // The constraint itself is X_ij, or (R^T R)_ij.
  return std::abs(dot(coordinates.col(i), coordinates.col(j)));
}

void LovaszThetaSDP::GradientConstraint(const size_t index,
                                        const arma::mat& coordinates,
                                        arma::mat& gradient)
{
//  Log::Debug << "Gradient of constraint " << index << " is " << std::endl;
  if (index == 0) // This is the constraint Tr(X) = 1.
  {
    gradient = 2 * coordinates; // d/dR (Tr(R R^T)) = 2 R.
//    std::cout << gradient;
    return;
  }

//  Log::Debug << "Evaluating gradient of constraint " << index << " with ";
  size_t i = edges(0, index - 1);
  size_t j = edges(1, index - 1);
//  Log::Debug << "i = " << i << " and j = " << j << "." << std::endl;

  // Since the constraint is (R^T R)_ij, the gradient for (x, y) will be (I
  // derived this for one of the MVU constraints):
  //   0     , y != i, y != j
  //   2 R_xj, y  = i, y != j
  //   2 R_xi, y != i, y  = j
  //   4 R_xy, y  = i, y  = j
  // This results in the gradient matrix having two nonzero rows; for row
  // i, the elements are R_nj, where n is the row; for column j, the elements
  // are R_ni.
  gradient.zeros(coordinates.n_rows, coordinates.n_cols);

  gradient.col(i) = coordinates.col(j);
  gradient.col(j) += coordinates.col(i); // In case j = i (shouldn't happen).

//  std::cout << gradient;
}

const arma::mat& LovaszThetaSDP::GetInitialPoint()
{
  if (initialPoint.n_rows != 0 && initialPoint.n_cols != 0)
    return initialPoint; // It has already been calculated.

//  Log::Debug << "Calculating initial point." << std::endl;

  // First, we must calculate the correct value of r.  The matrix we return, R,
  // will be r x V, because X = R^T R is of dimension V x V.
  // The rule for calculating r (from Monteiro and Burer, eq. 5) is
  //    r = max(r >= 0 : r (r + 1) / 2 <= m }
  // where m is equal to the number of constraints plus one.
  //
  // Solved, this is
  //   0.5 r^2 + 0.5 r - m = 0
  // which becomes
  //   r = (-0.5 [+/-] sqrt((-0.5)^2 - 4 * -0.5 * m)) / -1
  //   r = 0.5 [+/-] sqrt(0.25 + 2 m)
  // and because m is always positive,
  //   r = 0.5 + sqrt(0.25 + 2m)
  float m = NumConstraints();
  float r = 0.5 + sqrt(0.25 + 2 * m);
  if (ceil(r) > vertices)
    r = vertices; // An upper bound on the dimension.

  initialPoint.set_size(ceil(r), vertices);

  // Now we set the entries of the initial matrix according to the formula given
  // in Section 4 of Monteiro and Burer.
  for (size_t i = 0; i < r; i++)
  {
    for (size_t j = 0; j < (size_t) vertices; j++)
    {
      if (i == j)
        initialPoint(i, j) = sqrt(1.0 / r) + sqrt(1.0 / (vertices * m));
      else
        initialPoint(i, j) = sqrt(1.0 / (vertices * m));
    }
  }

  return initialPoint;
}
