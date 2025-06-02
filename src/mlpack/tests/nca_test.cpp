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
#include <mlpack/methods/nca.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace ens;

//
// Tests for the SoftmaxErrorFunction
//

/**
 * The Softmax error function should return the identity matrix as its initial
 * point.
 */
TEMPLATE_TEST_CASE("SoftmaxInitialPoint", "[NCATest]", float, double)
{
  using eT = TestType;

  // Cheap fake dataset.
  arma::Mat<eT> data;
  data.randu(5, 5);
  arma::Row<size_t> labels;
  labels.zeros(5);

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  // Verify the initial point is the identity matrix.
  arma::Mat<eT> initialPoint = sef.GetInitialPoint();
  const double eps = std::is_same_v<eT, float> ? 1e-4 : 1e-7;
  const double margin = std::is_same_v<eT, float> ? 1e-4 : 1e-5;
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      if (row == col)
        REQUIRE(initialPoint(row, col) == Approx(1.0).epsilon(eps));
      else
        REQUIRE(initialPoint(row, col) == Approx(0.0).margin(margin));
    }
  }
}

/***
 * On a simple fake dataset, ensure that the initial function evaluation is
 * correct.
 */
TEMPLATE_TEST_CASE("SoftmaxInitialEvaluation", "[NCATest]", float, double)
{
  using eT = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<eT> data       = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  eT objective = sef.Evaluate(arma::eye<arma::Mat<eT>>(2, 2));

  // Result painstakingly calculated by hand by rcurtin (recorded forever in his
  // notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  REQUIRE(objective == Approx(-1.5115).epsilon(0.0001));
}

/**
 * On a simple fake dataset, ensure that the initial gradient evaluation is
 * correct.
 */
TEMPLATE_TEST_CASE("SoftmaxInitialGradient", "[NCATest]", float, double)
{
  using eT = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<eT> data       = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  arma::Mat<eT> gradient;
  arma::Mat<eT> coordinates(2, 2, arma::fill::eye);
  sef.Gradient(coordinates, gradient);

  // Results painstakingly calculated by hand by rcurtin (recorded forever in
  // his notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  //
  // UPDATE 2024: that notebook definitely got thrown away over a decade ago.  I
  // don't even remember what it looked like.
  REQUIRE(gradient(0, 0) == Approx(-0.089766).epsilon(0.0005));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 1) == Approx(1.63823).epsilon(0.0001));
}

/**
 * On optimally separated datasets, ensure that the objective function is
 * optimal (equal to the negative number of points).
 */
TEMPLATE_TEST_CASE("SoftmaxOptimalEvaluation", "[NCATest]", float, double)
{
  using eT = TestType;

  // Simple optimal dataset.
  arma::Mat<eT> data       = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  eT objective = sef.Evaluate(arma::eye<arma::Mat<eT>>(2, 2));

  // Use a very close tolerance for optimality; we need to be sure this function
  // gives optimal results correctly.
  const double eps = std::is_same_v<eT, float> ? 1e-6 : 1e-12;
  REQUIRE(objective == Approx(-4.0).epsilon(eps));
}

/**
 * On optimally separated datasets, ensure that the gradient is zero.
 */
TEMPLATE_TEST_CASE("SoftmaxOptimalGradient", "[NCATest]", float, double)
{
  using eT = TestType;

  // Simple optimal dataset.
  arma::Mat<eT> data       = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  arma::Mat<eT> gradient;
  sef.Gradient(arma::eye<arma::Mat<eT>>(2, 2), gradient);

  REQUIRE(gradient(0, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(1e-5));
  REQUIRE(gradient(1, 1) == Approx(0.0).margin(1e-5));
}

/**
 * Ensure the separable objective function is right.
 */
TEMPLATE_TEST_CASE("SoftmaxSeparableObjective", "[NCATest]", float, double)
{
  using eT = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<eT> data       = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  // Results painstakingly calculated by hand by rcurtin (recorded forever in
  // his notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  arma::Mat<eT> coordinates = arma::eye<arma::Mat<eT>>(2, 2);
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
TEMPLATE_TEST_CASE("OptimalSoftmaxSeparableObjective", "[NCATest]", float,
    double)
{
  using eT = TestType;

  // Simple optimal dataset.
  arma::Mat<eT> data       = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  arma::Mat<eT> coordinates = arma::eye<arma::Mat<eT>>(2, 2);

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
TEMPLATE_TEST_CASE("SoftmaxSeparableGradient", "[NCATest]", float, double)
{
  using eT = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<eT> data       = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  arma::Mat<eT> coordinates = arma::eye<arma::Mat<eT>>(2, 2);
  arma::Mat<eT> gradient(2, 2);

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
TEMPLATE_TEST_CASE("NCASGDSimpleDataset", "[NCATest]", float, double)
{
  using eT = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<eT> data       = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Huge learning rate because this is so simple.
  ens::StandardSGD opt;
  opt.StepSize() = 1.2;
  opt.MaxIterations() = 300000;
  opt.Tolerance() = 0;
  opt.Shuffle() = true;

  arma::Mat<eT> outputMatrix;
  NCA nca;
  nca.LearnDistance(data, labels, outputMatrix, opt);

  // Ensure that the objective function is better now.
  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  eT initObj = sef.Evaluate(arma::eye<arma::Mat<eT>>(2, 2));
  eT finalObj = sef.Evaluate(outputMatrix);
  arma::Mat<eT> finalGradient;
  sef.Gradient(outputMatrix, finalGradient);

  // finalObj must be less than initObj.
  REQUIRE(finalObj < initObj);
  // Verify that final objective is optimal.
  REQUIRE(finalObj == Approx(-6.0).epsilon(0.00005));
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  REQUIRE(arma::norm(finalGradient, 2) < 1e-4);
}

TEMPLATE_TEST_CASE("NCALBFGSSimpleDataset", "[NCATest]", float, double)
{
  using eT = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<eT> data       = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  L_BFGS lbfgs;
  lbfgs.NumBasis() = 5;

  arma::Mat<eT> outputMatrix;
  NCA nca;
  nca.LearnDistance(data, labels, outputMatrix, lbfgs);

  // Ensure that the objective function is better now.
  SoftmaxErrorFunction<arma::Mat<eT>, arma::Row<size_t>,
      SquaredEuclideanDistance> sef(data, labels);

  eT initObj = sef.Evaluate(arma::eye<arma::Mat<eT>>(2, 2));
  eT finalObj = sef.Evaluate(outputMatrix);
  arma::Mat<eT> finalGradient;
  sef.Gradient(outputMatrix, finalGradient);

  // finalObj must be less than initObj.
  REQUIRE(finalObj < initObj);
  // Verify that final objective is optimal.
  REQUIRE(finalObj == Approx(-6.0).epsilon(0.00001));
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  REQUIRE(arma::norm(finalGradient, 2) < 1e-5);
}
