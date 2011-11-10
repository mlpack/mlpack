/**
 * @file nca_test.cpp
 * @author Ryan Curtin
 *
 * Unit tests for Neighborhood Components Analysis and related code (including
 * the softmax error function).
 */
#include <mlpack/core.h>
#include <mlpack/core/kernels/lmetric.hpp>
#include <mlpack/methods/nca/nca.h>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::nca;

//
// Tests for the SoftmaxErrorFunction
//

BOOST_AUTO_TEST_SUITE(NCATest);

/**
 * The Softmax error function should return the identity matrix as its initial
 * point.
 */
BOOST_AUTO_TEST_CASE(softmax_initial_point)
{
  // Cheap fake dataset.
  arma::mat data;
  data.randu(5, 5);
  arma::uvec labels;
  labels.zeros(5);

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  // Verify the initial point is the identity matrix.
  arma::mat initial_point = sef.GetInitialPoint();
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      if (row == col)
        BOOST_REQUIRE_CLOSE(initial_point(row, col), 1.0, 1e-5);
      else
        BOOST_REQUIRE(initial_point(row, col) == 0.0);
    }
  }
}

/***
 * On a simple fake dataset, ensure that the initial function evaluation is
 * correct.
 */
BOOST_AUTO_TEST_CASE(softmax_initial_evaluation)
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
BOOST_AUTO_TEST_CASE(softmax_initial_gradient)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data    = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                      " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::uvec labels = " 0    0    0    1    1    1   ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat gradient;
  sef.Gradient(arma::eye<arma::mat>(2, 2), gradient);

  // Results painstakingly calculated by hand by rcurtin (recorded forever in
  // his notebook).  As a result of lack of precision of the by-hand result, the
  // tolerance is fairly high.
  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.089766, 0.05);
  BOOST_REQUIRE(gradient(1, 0) == 0.0);
  BOOST_REQUIRE(gradient(0, 1) == 0.0);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 1.63823, 0.01);
}

/**
 * On optimally separated datasets, ensure that the objective function is
 * optimal (equal to the negative number of points).
 */
BOOST_AUTO_TEST_CASE(softmax_optimal_evaluation)
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
BOOST_AUTO_TEST_CASE(softmax_optimal_gradient)
{
  // Simple optimal dataset.
  arma::mat data    = " 500  500 -500 -500;"
                      "   1    0    1    0 ";
  arma::uvec labels = "   0    0    1    1 ";

  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  arma::mat gradient;
  sef.Gradient(arma::eye<arma::mat>(2, 2), gradient);

  BOOST_REQUIRE(gradient(0, 0) == 0.0);
  BOOST_REQUIRE(gradient(0, 1) == 0.0);
  BOOST_REQUIRE(gradient(1, 0) == 0.0);
  BOOST_REQUIRE(gradient(1, 1) == 0.0);
}

//
// Tests for the NCA algorithm.
//

/**
 * On our simple dataset, ensure that the NCA algorithm fully separates the
 * points.
 */
BOOST_AUTO_TEST_CASE(nca_simple_dataset)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat data    = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                      " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::uvec labels = " 0    0    0    1    1    1   ";

  NCA<SquaredEuclideanDistance> nca(data, labels);

  arma::mat output_matrix;
  nca.LearnDistance(output_matrix);

  // Ensure that the objective function is better now.
  SoftmaxErrorFunction<SquaredEuclideanDistance> sef(data, labels);

  double init_obj = sef.Evaluate(arma::eye<arma::mat>(2, 2));
  double final_obj = sef.Evaluate(output_matrix);
  arma::mat final_gradient;
  sef.Gradient(output_matrix, final_gradient);

  // final_obj must be less than init_obj.
  BOOST_REQUIRE_LT(final_obj, init_obj);
  // Verify that final objective is optimal.
  BOOST_REQUIRE_CLOSE(final_obj, -6.0, 1e-8);
  // The solution is not unique, so the best we can do is ensure the gradient
  // norm is close to 0.
  BOOST_REQUIRE_LT(arma::norm(final_gradient, 2), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END();
