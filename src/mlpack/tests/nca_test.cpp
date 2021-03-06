/**
 * @file tests/nca_test.cpp
 * @author Ryan Curtin
 *
 * Unit tests for Neighborhood Components Analysis and related code (including
 * the softmax error function).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/nca/nca.hpp>
#include <ensmallen.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::nca;
using namespace ens;

//
// Tests for the SoftmaxErrorFunction
//

/**
 * The Softmax error function should return the identity matrix as its initial
 * point.
 */
TEST_CASE("SoftmaxInitialPoint", "[NCATesT]")
{
  // Cheap fake dataset.
  arma::mat data;
  data.randu(5, 5);
  arma::Row<size_t> labels;
  labels.zeros(5);

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  // Verify the initial point is the identity matrix.
  arma::mat initialPoint = sef.GetInitialPoint();
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      if (row == col)
        REQUIRE(initialPoint(row, col) == Approx(1.0).epsilon(1e-7));
      else
        REQUIRE(initialPoint(row, col) == Approx(0.0).margin(1e-5));
    }
  }
}

/***
 * On a simple fake dataset, ensure that the initial function evaluation is
 * correct.
 */
TEST_CASE("SoftmaxInitialEvaluation", "[NCATesT]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  double objective = sef.Evaluate(arma::eye<arma::mat>(2, 2));

  // Result painstakingly calculated by hand by rcurtin (recorded forever in his
  // notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  REQUIRE(objective == Approx(-1.5115).epsilon(0.0001));
}

/**
 * On a simple fake dataset, ensure that the initial gradient evaluation is
 * correct.
 */
TEST_CASE("SoftmaxInitialGradient", "[NCATesT]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat gradient;
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  sef.Gradient(coordinates, gradient);

  // Results painstakingly calculated by hand by rcurtin (recorded forever in
  // his notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  REQUIRE(gradient(0, 0) == Approx(-0.089766).epsilon(0.0005));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 1) == Approx(1.63823).epsilon(0.0001));
}

/**
 * On optimally separated datasets, ensure that the objective function is
 * optimal (equal to the negative number of points).
 */
TEST_CASE("SoftmaxOptimalEvaluation", "[NCATesT]")
{
  // Simple optimal dataset.
  arma::mat data           = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  double objective = sef.Evaluate(arma::eye<arma::mat>(2, 2));

  // Use a very close tolerance for optimality; we need to be sure this function
  // gives optimal results correctly.
  REQUIRE(objective == Approx(-4.0).epsilon(1e-12));
}

/**
 * On optimally separated datasets, ensure that the gradient is zero.
 */
TEST_CASE("SoftmaxOptimalGradient", "[NCATesT]")
{
  // Simple optimal dataset.
  arma::mat data           = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat gradient;
  sef.Gradient(arma::eye<arma::mat>(2, 2), gradient);

  REQUIRE(gradient(0, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 1) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure the separable objective function is right.
 */
TEST_CASE("SoftmaxSeparableObjective", "[NCATesT]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  // Results painstakingly calculated by hand by rcurtin (recorded forever in
  // his notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  REQUIRE(sef.Evaluate(coordinates, 0, 1) == Approx(-0.22480).epsilon(0.0001));
  REQUIRE(sef.Evaluate(coordinates, 1, 1) == Approx(-0.30613).epsilon(0.0001));
  REQUIRE(sef.Evaluate(coordinates, 2, 1) == Approx(-0.22480).epsilon(0.0001));
  REQUIRE(sef.Evaluate(coordinates, 3, 1) == Approx(-0.22480).epsilon(0.0001));
  REQUIRE(sef.Evaluate(coordinates, 4, 1) == Approx(-0.30613).epsilon(0.0001));
  REQUIRE(sef.Evaluate(coordinates, 5, 1) == Approx(-0.22480).epsilon(0.0001));
}

/**
 * Ensure the optimal separable objective function is right.
 */
TEST_CASE("OptimalSoftmaxSeparableObjective", "[NCATesT]")
{
  // Simple optimal dataset.
  arma::mat data           = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);

  // Use a very close tolerance for optimality; we need to be sure this function
  // gives optimal results correctly.
  REQUIRE(sef.Evaluate(coordinates, 0, 1) == Approx(-1.0).epsilon(1e-12));
  REQUIRE(sef.Evaluate(coordinates, 1, 1) == Approx(-1.0).epsilon(1e-12));
  REQUIRE(sef.Evaluate(coordinates, 2, 1) == Approx(-1.0).epsilon(1e-12));
  REQUIRE(sef.Evaluate(coordinates, 3, 1) == Approx(-1.0).epsilon(1e-12));
}

/**
 * Ensure the separable gradient is right.
 */
TEST_CASE("SoftmaxSeparableGradient", "[NCATesT]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  arma::mat gradient(2, 2);

  sef.Gradient(coordinates, 0, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
  REQUIRE(gradient(0, 1) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 0) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.14359).epsilon(0.0001));

  sef.Gradient(coordinates, 1, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.008496).epsilon(0.0001));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.12238).epsilon(0.0001));

  sef.Gradient(coordinates, 2, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
  REQUIRE(gradient(0, 1) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 0) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.1435886).epsilon(0.0001));

  sef.Gradient(coordinates, 3, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
  REQUIRE(gradient(0, 1) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 0) == Approx(-2.0 * 0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.1435886).epsilon(0.0001));

  sef.Gradient(coordinates, 4, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.008496).epsilon(0.0001));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.12238).epsilon(0.0001));

  sef.Gradient(coordinates, 5, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-2.0 * 0.0069708).epsilon(0.0001));
  REQUIRE(gradient(0, 1) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 0) == Approx(-2.0 * -0.0101707).epsilon(0.0001));
  REQUIRE(gradient(1, 1) == Approx(-2.0 * -0.1435886).epsilon(0.0001));
}

//
// Tests for the NCA algorithm.
//

/**
 * On our simple dataset, ensure that the NCA algorithm fully separates the
 * points.
 */
TEST_CASE("NCASGDSimpleDataset", "[NCATesT]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Huge learning rate because this is so simple.
  NCA<SquaredEuclideanDistance> nca(data, labels);
  nca.Optimizer().StepSize() = 1.2;
  nca.Optimizer().MaxIterations() = 300000;
  nca.Optimizer().Tolerance() = 0;
  nca.Optimizer().Shuffle() = true;

  arma::mat outputMatrix;
  nca.LearnDistance(outputMatrix);

  // Ensure that the objective function is better now.
  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  double initObj = sef.Evaluate(arma::eye<arma::mat>(2, 2));
  double finalObj = sef.Evaluate(outputMatrix);
  arma::mat finalGradient;
  sef.Gradient(outputMatrix, finalGradient);

  // finalObj must be less than initObj.
  REQUIRE(finalObj < initObj);
  // Verify that final objective is optimal.
  REQUIRE(finalObj == Approx(-6.0).epsilon(0.00005));
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  REQUIRE(arma::norm(finalGradient, 2) < 1e-4);
}

TEST_CASE("NCALBFGSSimpleDataset", "[NCATesT]")
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Huge learning rate because this is so simple.
  NCA<SquaredEuclideanDistance, L_BFGS> nca(data, labels);
  nca.Optimizer().NumBasis() = 5;

  arma::mat outputMatrix;
  nca.LearnDistance(outputMatrix);

  // Ensure that the objective function is better now.
  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  double initObj = sef.Evaluate(arma::eye<arma::mat>(2, 2));
  double finalObj = sef.Evaluate(outputMatrix);
  arma::mat finalGradient;
  sef.Gradient(outputMatrix, finalGradient);

  // finalObj must be less than initObj.
  REQUIRE(finalObj < initObj);
  // Verify that final objective is optimal.
  REQUIRE(finalObj == Approx(-6.0).epsilon(1e-7));
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  REQUIRE(arma::norm(finalGradient, 2) < 1e-6);
}
