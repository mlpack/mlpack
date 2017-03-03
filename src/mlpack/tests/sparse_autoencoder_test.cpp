/**
 * @file sparse_autoencoder_test.cpp
 * @author Siddharth Agrawal
 *
 * Test the SparseAutoencoder class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::nn;

BOOST_AUTO_TEST_SUITE(SparseAutoencoderTest);

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionEvaluate)
{
  const size_t vSize = 5;
  const size_t hSize = 3;
  const size_t r = 2 * hSize + 1;
  const size_t c = vSize + 1;

  // Simple fake dataset.
  arma::mat data1("0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5");
  // Transpose of the above dataset.
  arma::mat data2 = data1.t();

  // Create a SparseAutoencoderFunction. Regularization and KL divergence terms
  // ignored.
  SparseAutoencoderFunction saf1(data1, vSize, hSize, 0, 0);

  // Test using first dataset. Values were calculated using Octave.
  BOOST_REQUIRE_CLOSE(saf1.Evaluate(arma::ones(r, c)), 1.190472606540, 1e-5);
  BOOST_REQUIRE_CLOSE(saf1.Evaluate(arma::zeros(r, c)), 0.150000000000, 1e-5);
  BOOST_REQUIRE_CLOSE(saf1.Evaluate(-arma::ones(r, c)), 0.048800332266, 1e-5);

  // Create a SparseAutoencoderFunction. Regularization and KL divergence terms
  // ignored.
  SparseAutoencoderFunction saf2(data2, vSize, hSize, 0, 0);

  // Test using second dataset. Values were calculated using Octave.
  BOOST_REQUIRE_CLOSE(saf2.Evaluate(arma::ones(r, c)), 1.197585812647, 1e-5);
  BOOST_REQUIRE_CLOSE(saf2.Evaluate(arma::zeros(r, c)), 0.150000000000, 1e-5);
  BOOST_REQUIRE_CLOSE(saf2.Evaluate(-arma::ones(r, c)), 0.063466617408, 1e-5);
}

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionRandomEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t vSize = 20;
  const size_t hSize = 10;
  const size_t l1 = hSize;
  const size_t l2 = vSize;
  const size_t l3 = 2 * hSize;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(vSize, points);

  // Create a SparseAutoencoderFunction. Regularization and KL divergence terms
  // ignored.
  SparseAutoencoderFunction saf(data, vSize, hSize, 0, 0);

  // Run a number of trials.
  for (size_t i = 0; i < trials; i++)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(l3 + 1, l2 + 1);

    double reconstructionError = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; j++)
    {
      arma::mat hiddenLayer, outputLayer, diff;

      hiddenLayer = 1.0 /
          (1 + arma::exp(-(parameters.submat(0, 0, l1 - 1, l2 - 1) *
          data.col(j) + parameters.submat(0, l2, l1 - 1, l2))));
      outputLayer = 1.0 /
          (1 + arma::exp(-(parameters.submat(l1, 0, l3 - 1,l2 - 1).t()
          * hiddenLayer + parameters.submat(l3, 0, l3, l2 - 1).t())));
      diff = outputLayer - data.col(j);

      reconstructionError += 0.5 * arma::sum(arma::sum(diff % diff));
    }
    reconstructionError /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(saf.Evaluate(parameters), reconstructionError, 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionRegularizationEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t vSize = 20;
  const size_t hSize = 10;
  const size_t l2 = vSize;
  const size_t l3 = 2 * hSize;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(vSize, points);

  // 3 objects for comparing regularization costs.
  SparseAutoencoderFunction safNoReg(data, vSize, hSize, 0, 0);
  SparseAutoencoderFunction safSmallReg(data, vSize, hSize, 0.5, 0);
  SparseAutoencoderFunction safBigReg(data, vSize, hSize, 20, 0);

  // Run a number of trials.
  for (size_t i = 0; i < trials; i++)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(l3 + 1, l2 + 1);

    double wL2SquaredNorm;

    wL2SquaredNorm = arma::accu(parameters.submat(0, 0, l3 - 1, l2 - 1) %
        parameters.submat(0, 0, l3 - 1, l2 - 1));

    // Calculate regularization terms.
    const double smallRegTerm = 0.25 * wL2SquaredNorm;
    const double bigRegTerm = 10 * wL2SquaredNorm;

    BOOST_REQUIRE_CLOSE(safNoReg.Evaluate(parameters) + smallRegTerm,
        safSmallReg.Evaluate(parameters), 1e-5);
    BOOST_REQUIRE_CLOSE(safNoReg.Evaluate(parameters) + bigRegTerm,
        safBigReg.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionKLDivergenceEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t vSize = 20;
  const size_t hSize = 10;
  const size_t l1 = hSize;
  const size_t l2 = vSize;
  const size_t l3 = 2 * hSize;

  const double rho = 0.01;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(vSize, points);

  // 3 objects for comparing divergence costs.
  SparseAutoencoderFunction safNoDiv(data, vSize, hSize, 0, 0, rho);
  SparseAutoencoderFunction safSmallDiv(data, vSize, hSize, 0, 5, rho);
  SparseAutoencoderFunction safBigDiv(data, vSize, hSize, 0, 20, rho);

  // Run a number of trials.
  for(size_t i = 0; i < trials; i++)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(l3 + 1, l2 + 1);

    arma::mat rhoCap;
    rhoCap.zeros(hSize, 1);

    // Compute hidden layer activations for each example.
    for (size_t j = 0; j < points; j++)
    {
      arma::mat hiddenLayer;

      hiddenLayer = 1.0 / (1 +
          arma::exp(-(parameters.submat(0, 0, l1 - 1, l2 - 1) *
          data.col(j) + parameters.submat(0, l2, l1 - 1, l2))));
      rhoCap += hiddenLayer;
    }
    rhoCap /= points;

    // Calculate divergence terms.
    const double smallDivTerm = 5 * arma::accu(rho * arma::log(rho / rhoCap) +
        (1 - rho) * arma::log((1 - rho) / (1 - rhoCap)));
    const double bigDivTerm = 20 * arma::accu(rho * arma::log(rho / rhoCap) +
        (1 - rho) * arma::log((1 - rho) / (1 - rhoCap)));

    BOOST_REQUIRE_CLOSE(safNoDiv.Evaluate(parameters) + smallDivTerm,
        safSmallDiv.Evaluate(parameters), 1e-5);
    BOOST_REQUIRE_CLOSE(safNoDiv.Evaluate(parameters) + bigDivTerm,
        safBigDiv.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionGradient)
{
  const size_t points = 1000;
  const size_t vSize = 20;
  const size_t hSize = 10;
  const size_t l2 = vSize;
  const size_t l3 = 2 * hSize;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(vSize, points);

  // 3 objects for 3 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  SparseAutoencoderFunction saf1(data, vSize, hSize, 0, 0);
  SparseAutoencoderFunction saf2(data, vSize, hSize, 20, 0);
  SparseAutoencoderFunction saf3(data, vSize, hSize, 20, 20);

  // Create a random set of parameters.
  arma::mat parameters;
  parameters.randu(l3 + 1, l2 + 1);

  // Get gradients for the current parameters.
  arma::mat gradient1, gradient2, gradient3;
  saf1.Gradient(parameters, gradient1);
  saf2.Gradient(parameters, gradient2);
  saf3.Gradient(parameters, gradient3);

  // Perturbation constant.
  const double epsilon = 0.0001;
  double costPlus1, costMinus1, numGradient1;
  double costPlus2, costMinus2, numGradient2;
  double costPlus3, costMinus3, numGradient3;

  // For each parameter.
  for (size_t i = 0; i <= l3; i++)
  {
    for (size_t j = 0; j <= l2; j++)
    {
      // Perturb parameter with a positive constant and get costs.
      parameters(i, j) += epsilon;
      costPlus1 = saf1.Evaluate(parameters);
      costPlus2 = saf2.Evaluate(parameters);
      costPlus3 = saf3.Evaluate(parameters);

      // Perturb parameter with a negative constant and get costs.
      parameters(i, j) -= 2 * epsilon;
      costMinus1 = saf1.Evaluate(parameters);
      costMinus2 = saf2.Evaluate(parameters);
      costMinus3 = saf3.Evaluate(parameters);

      // Compute numerical gradients using the costs calculated above.
      numGradient1 = (costPlus1 - costMinus1) / (2 * epsilon);
      numGradient2 = (costPlus2 - costMinus2) / (2 * epsilon);
      numGradient3 = (costPlus3 - costMinus3) / (2 * epsilon);

      // Restore the parameter value.
      parameters(i, j) += epsilon;

      // Compare numerical and backpropagation gradient values.
      BOOST_REQUIRE_CLOSE(numGradient1, gradient1(i, j), 1e-2);
      BOOST_REQUIRE_CLOSE(numGradient2, gradient2(i, j), 1e-2);
      BOOST_REQUIRE_CLOSE(numGradient3, gradient3(i, j), 1e-2);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
