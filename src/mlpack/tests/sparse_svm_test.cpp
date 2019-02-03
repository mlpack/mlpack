/**
 * @file sparse_svm_test.cpp
 * @author Ayush Chamoli
 *
 * Test the Sparse SVM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_svm/sparse_svm.hpp>
#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(SparseSVMTest);

/**
 * A more complicated test for the SparseSVMFunction.
 */
BOOST_AUTO_TEST_CASE(SparseSVMFunctionEvaluate)
{
  const size_t dimension = 10;
  const size_t points = 1000;
  const size_t trials = 50;

  // Initialize a random dataset.
  arma::sp_mat data;
  data = arma::sprandu(dimension, points, 0.2);

  // Create random response.
  arma::Row<size_t> responses(points);
  for (size_t i = 0; i < points; ++i)
    responses(i) = math::RandInt(0, 2);

  // Create a SparseSVMFunction.
  SparseSVMFunction<> svm(data, responses, 0.0);

  // Run a bunch of trials.
  for (size_t i = 0; i < trials; i++)
  {
    // Generate a random set of parameters.
    arma::mat parameters;
    parameters = arma::randu<arma::mat>(1, dimension + 1);

    // Hand-calculate the Hinge Loss Function.
    double hingeloss = 0.0;
    for (size_t j = 0; j < points; j++)
    {
      hingeloss += std::max(0.0, 1 - (2 * (double)responses(j) - 1) *
          (arma::dot(data.col(j), parameters.head_cols(parameters.n_cols - 1))
          + parameters(parameters.n_cols - 1)));
    }
    hingeloss /= points;

    BOOST_REQUIRE_CLOSE(svm.Evaluate(parameters), hingeloss, 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
