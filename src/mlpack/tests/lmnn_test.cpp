/**
 * @file lmnn_test.cpp
 * @author Manish Kumar
 * Unit tests for Large Margin Nearest Neighbors and related code (including
 * the constraints class).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::lmnn;
using namespace mlpack::optimization;


BOOST_AUTO_TEST_SUITE(LMNNTest);

//
// Tests for the Constraints.
//

/**
 * The target neighbors function should be correct.
 * point.
 */
BOOST_AUTO_TEST_CASE(LMNNTargetNeighborsTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  Constraints constraint(dataset, labels, 1);

  //! Store target neighbors of data points.
  arma::Mat<size_t> targetNeighbors;

  constraint.TargetNeighbors(targetNeighbors);

  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 0), 1);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 1), 0);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 2), 1);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 3), 4);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 4), 3);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 5), 4);
}

/**
 * The impostors function should be correct.
 */
BOOST_AUTO_TEST_CASE(LMNNImpostorsTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  Constraints constraint(dataset, labels, 1);

  //! Store impostors of data points.
  arma::Mat<size_t> impostors;

  constraint.Impostors(impostors);

  BOOST_REQUIRE_EQUAL(impostors(0, 0), 3);
  BOOST_REQUIRE_EQUAL(impostors(0, 1), 4);
  BOOST_REQUIRE_EQUAL(impostors(0, 2), 5);
  BOOST_REQUIRE_EQUAL(impostors(0, 3), 0);
  BOOST_REQUIRE_EQUAL(impostors(0, 4), 1);
  BOOST_REQUIRE_EQUAL(impostors(0, 5), 2);
}

//
// Tests for the LMNNFunction
//

/**
 * The LMNN function should return the identity matrix as its initial
 * point.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialPointTest)
{
  // Cheap fake dataset.
  arma::mat dataset = arma::randu(5, 5);
  arma::Row<size_t> labels = "0 1 1 0 0";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.5);

  // Verify the initial point is the identity matrix.
  arma::mat initialPoint = lmnnfn.GetInitialPoint();
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
 * Ensure non-seprable objective function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialEvaluation)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  double objective = lmnnfn.Evaluate(arma::eye<arma::mat>(2, 2));

  // Result calculated by hand.
  BOOST_REQUIRE_CLOSE(objective, 9.456, 1e-5);
}

/**
 * Ensure non-seprable gradient function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialGradient)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  arma::mat gradient;
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  lmnnfn.Gradient(coordinates, gradient);

  // Result calculated by hand.
  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.288, 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 12.0, 1e-5);
}

/***
 * Ensure non-seprable EvaluateWithGradient function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialEvaluateWithGradient)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  arma::mat gradient;
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  double objective = lmnnfn.EvaluateWithGradient(coordinates, gradient);

  // Result calculated by hand.
  BOOST_REQUIRE_CLOSE(objective, 9.456, 1e-5);
  // Check Gradient
  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.288, 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 12.0, 1e-5);
}

/**
 * Ensure the separable objective function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNSeparableObjective)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  // Result calculated by hand.
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 0, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 1, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 2, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 3, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 4, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 5, 1), 1.576, 1e-5);
}

/**
 * Ensure the separable gradient is right.
 */
BOOST_AUTO_TEST_CASE(LMNNSeparableGradient)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  arma::mat gradient(2, 2);

  lmnnfn.Gradient(coordinates, 0, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 1, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 2, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 3, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 4, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 5, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);
}

/**
 * Ensure the separable EvaluateWithGradient function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNSeparableEvaluateWithGradient)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  arma::mat gradient(2, 2);

  double objective = lmnnfn.EvaluateWithGradient(coordinates, 0, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 1, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 2, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 3, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 4, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 5, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);
}

BOOST_AUTO_TEST_CASE(LMNNSGDSimpleDataset)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNN<> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  // Ensure that the objective function is better now.
  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  double initObj = lmnnfn.Evaluate(arma::eye<arma::mat>(2, 2));
  double finalObj = lmnnfn.Evaluate(outputMatrix);

  // finalObj must be less than initObj.
  BOOST_REQUIRE_LT(finalObj, initObj);
}

BOOST_AUTO_TEST_CASE(LMNNLBFGSSimpleDataset)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNN<SquaredEuclideanDistance, L_BFGS> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  // Ensure that the objective function is better now.
  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6);

  double initObj = lmnnfn.Evaluate(arma::eye<arma::mat>(2, 2));
  double finalObj = lmnnfn.Evaluate(outputMatrix);

  // finalObj must be less than initObj.
  BOOST_REQUIRE_LT(finalObj, initObj);
}

BOOST_AUTO_TEST_SUITE_END();
