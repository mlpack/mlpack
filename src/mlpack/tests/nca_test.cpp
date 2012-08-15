/**
 * @file nca_test.cpp
 * @author Ryan Curtin
 *
 * Unit tests for Neighborhood Components Analysis and related code (including
 * the softmax error function).
 * This file is part of MLPACK 1.0.2.
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
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/nca/nca.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::nca;

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
  arma::uvec labels;
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
  arma::mat data    = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                      " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::uvec labels = " 0    0    0    1    1    1   ";

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
  arma::mat data    = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                      " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::uvec labels = " 0    0    0    1    1    1   ";

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
  arma::mat data    = " 500  500 -500 -500;"
                      "   1    0    1    0 ";
  arma::uvec labels = "   0    0    1    1 ";

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
  arma::mat data    = " 500  500 -500 -500;"
                      "   1    0    1    0 ";
  arma::uvec labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat gradient;
  sef.Gradient(arma::eye<arma::mat>(2, 2), gradient);

  BOOST_REQUIRE_SMALL(gradient(0, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 1), 1e-5);
}

//
// Tests for the NCA algorithm.
//

/**
 * On our simple dataset, ensure that the NCA algorithm fully separates the
 * points.
 */
BOOST_AUTO_TEST_CASE(NcaSimpleDataset)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data    = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                      " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::uvec labels = " 0    0    0    1    1    1   ";

  NCA<SquaredEuclideanDistance> nca(data, labels);

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
  BOOST_REQUIRE_CLOSE(finalObj, -6.0, 1e-8);
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  BOOST_REQUIRE_LT(arma::norm(finalGradient, 2), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END();
