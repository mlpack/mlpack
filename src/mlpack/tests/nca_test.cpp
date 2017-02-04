/**
 * @file nca_test.cpp
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
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::nca;
using namespace mlpack::optimization;

//
// Tests for the SoftmaxErrorFunction
//

BOOST_AUTO_TEST_SUITE(NCATest);

/**
 * The Softmax error function should return the identity matrix as its initial
 * point.
 */
BOOST_AUTO_TEST_CASE(SoftmaxInitialPoint)
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
        BOOST_REQUIRE_CLOSE(initialPoint(row, col), 1.0, 1e-5);
      else
        BOOST_REQUIRE_SMALL(initialPoint(row, col), 1e-5);
    }
  }
}

/***
 * On a simple fake dataset, ensure that the initial function evaluation is
 * correct.
 */
BOOST_AUTO_TEST_CASE(SoftmaxInitialEvaluation)
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
  BOOST_REQUIRE_CLOSE(objective, -1.5115, 0.01);
}

/**
 * On a simple fake dataset, ensure that the initial gradient evaluation is
 * correct.
 */
BOOST_AUTO_TEST_CASE(SoftmaxInitialGradient)
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
  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.089766, 0.05);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 1.63823, 0.01);
}

/**
 * On optimally separated datasets, ensure that the objective function is
 * optimal (equal to the negative number of points).
 */
BOOST_AUTO_TEST_CASE(SoftmaxOptimalEvaluation)
{
  // Simple optimal dataset.
  arma::mat data           = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  double objective = sef.Evaluate(arma::eye<arma::mat>(2, 2));

  // Use a very close tolerance for optimality; we need to be sure this function
  // gives optimal results correctly.
  BOOST_REQUIRE_CLOSE(objective, -4.0, 1e-10);
}

/**
 * On optimally separated datasets, ensure that the gradient is zero.
 */
BOOST_AUTO_TEST_CASE(SoftmaxOptimalGradient)
{
  // Simple optimal dataset.
  arma::mat data           = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat gradient;
  sef.Gradient(arma::eye<arma::mat>(2, 2), gradient);

  BOOST_REQUIRE_SMALL(gradient(0, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 1), 1e-5);
}

/**
 * Ensure the separable objective function is right.
 */
BOOST_AUTO_TEST_CASE(SoftmaxSeparableObjective)
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
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 0), -0.22480, 0.01);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 1), -0.30613, 0.01);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 2), -0.22480, 0.01);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 3), -0.22480, 0.01);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 4), -0.30613, 0.01);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 5), -0.22480, 0.01);
}

/**
 * Ensure the optimal separable objective function is right.
 */
BOOST_AUTO_TEST_CASE(OptimalSoftmaxSeparableObjective)
{
  // Simple optimal dataset.
  arma::mat data           = " 500  500 -500 -500;"
                             "   1    0    1    0 ";
  arma::Row<size_t> labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);

  // Use a very close tolerance for optimality; we need to be sure this function
  // gives optimal results correctly.
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 0), -1.0, 1e-10);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 1), -1.0, 1e-10);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 2), -1.0, 1e-10);
  BOOST_REQUIRE_CLOSE(sef.Evaluate(coordinates, 3), -1.0, 1e-10);
}

/**
 * Ensure the separable gradient is right.
 */
BOOST_AUTO_TEST_CASE(SoftmaxSeparableGradient)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  arma::mat gradient(2, 2);

  sef.Gradient(coordinates, 0, gradient);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -2.0 * 0.0069708, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), -2.0 * -0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), -2.0 * -0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), -2.0 * -0.14359, 0.01);

  sef.Gradient(coordinates, 1, gradient);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -2.0 * 0.008496, 0.01);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), -2.0 * -0.12238, 0.01);

  sef.Gradient(coordinates, 2, gradient);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -2.0 * 0.0069708, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), -2.0 * 0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), -2.0 * 0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), -2.0 * -0.1435886, 0.01);

  sef.Gradient(coordinates, 3, gradient);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -2.0 * 0.0069708, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), -2.0 * 0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), -2.0 * 0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), -2.0 * -0.1435886, 0.01);

  sef.Gradient(coordinates, 4, gradient);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -2.0 * 0.008496, 0.01);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), -2.0 * -0.12238, 0.01);

  sef.Gradient(coordinates, 5, gradient);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -2.0 * 0.0069708, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), -2.0 * -0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), -2.0 * -0.0101707, 0.01);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), -2.0 * -0.1435886, 0.01);
}

//
// Tests for the NCA algorithm.
//

/**
 * On our simple dataset, ensure that the NCA algorithm fully separates the
 * points.
 */
BOOST_AUTO_TEST_CASE(NCASGDSimpleDataset)
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
  BOOST_REQUIRE_LT(finalObj, initObj);
  // Verify that final objective is optimal.
  BOOST_REQUIRE_CLOSE(finalObj, -6.0, 0.005);
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  BOOST_REQUIRE_LT(arma::norm(finalGradient, 2), 1e-4);
}

BOOST_AUTO_TEST_CASE(NCALBFGSSimpleDataset)
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
  BOOST_REQUIRE_LT(finalObj, initObj);
  // Verify that final objective is optimal.
  BOOST_REQUIRE_CLOSE(finalObj, -6.0, 1e-5);
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  BOOST_REQUIRE_LT(arma::norm(finalGradient, 2), 1e-6);

}

BOOST_AUTO_TEST_SUITE_END();
