/**
 * @file sparse_autoencoder_test.cpp
 * @author Siddharth Agrawal
 * @author Tham Ngap Wei
 *
 * Test the SparseAutoencoder class.
 */
#include <mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp>

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;

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
  SparseAutoencoder<> saf1(data1, vSize, hSize, 0, 0);

  // Test using first dataset. Values were calculated using Octave.
  BOOST_REQUIRE_CLOSE(saf1.Evaluate(arma::ones(r, c)), 1.190472606540, 1e-5);
  BOOST_REQUIRE_CLOSE(saf1.Evaluate(arma::zeros(r, c)), 0.150000000000, 1e-5);
  BOOST_REQUIRE_CLOSE(saf1.Evaluate(-arma::ones(r, c)), 0.048800332266, 1e-5);

  // Create a SparseAutoencoderFunction. Regularization and KL divergence terms
  // ignored.
  SparseAutoencoder<> saf2(data2, vSize, hSize, 0, 0);

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
  SparseAutoencoder<> saf(data, vSize, hSize, 0, 0);

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
          (1 + arma::exp(-(parameters.submat(l1, 0, l3 - 1, l2 - 1).t()
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
  SparseAutoencoder<> safNoReg(data, vSize, hSize, 0, 0);
  SparseAutoencoder<> safSmallReg(data, vSize, hSize, 0.5, 0);
  SparseAutoencoder<> safBigReg(data, vSize, hSize, 20, 0);

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
  SparseAutoencoder<> safNoDiv(data, vSize, hSize, 0, 0, rho);
  SparseAutoencoder<> safSmallDiv(data, vSize, hSize, 0, 5, rho);
  SparseAutoencoder<> safBigDiv(data, vSize, hSize, 0, 20, rho);

  // Run a number of trials.
  for (size_t i = 0; i < trials; i++)
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

BOOST_AUTO_TEST_SUITE_END();
