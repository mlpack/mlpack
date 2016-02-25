/**
 * @file rmsprop_test.cpp
 * @author Marcus Edel
 *
 * Tests the RMSProp optimizer.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(RMSpropTest);

/**
 * Tests the RMSprop optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleRMSpropTestFunction)
{
  SGDTestFunction f;
  RMSprop<SGDTestFunction> optimizer(f, 1e-3, 0.99, 1e-8, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  const double result = optimizer.Optimize(coordinates);

  BOOST_REQUIRE_LE(std::abs(result) - 1.0, 0.2);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-3);
}

/**
 * Run RMSprop on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(LogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  LogisticRegression<> lr(shuffledData.n_rows, 0.5);

  LogisticRegressionFunction<> lrf(shuffledData, shuffledResponses, 0.5);
  RMSprop<LogisticRegressionFunction<> > rmsprop(lrf);
  lr.Train(rmsprop);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Run RMSprop on a feedforward neural network and make sure the results are
 * acceptable.
 */
BOOST_AUTO_TEST_CASE(FeedforwardTest)
{
  // Test on a non-linearly separable dataset (XOR).
  arma::mat input, labels;
  input << 0 << 1 << 1 << 0 << arma::endr
        << 1 << 0 << 1 << 0 << arma::endr;
  labels << 0 << 0 << 1 << 1;

  // Instantiate the first layer.
  LinearLayer<> inputLayer(input.n_rows, 4);
  BiasLayer<> biasLayer(4);
  SigmoidLayer<> hiddenLayer0;

  // Instantiate the second layer.
  LinearLayer<> hiddenLayer1(4, labels.n_rows);
  SigmoidLayer<> outputLayer;

  // Instantiate the output layer.
  BinaryClassificationLayer classOutputLayer;

  // Instantiate the feedforward network.
  auto modules = std::tie(inputLayer, biasLayer, hiddenLayer0, hiddenLayer1,
      outputLayer);
  FFN<decltype(modules), decltype(classOutputLayer), RandomInitialization,
      MeanSquaredErrorFunction> net(modules, classOutputLayer);

  RMSprop<decltype(net)> opt(net, 0.1, 0.88, 1e-15,
      300 * input.n_cols, 1e-18);

  net.Train(input, labels, opt);

  arma::mat prediction;
  net.Predict(input, prediction);

  const bool b = arma::accu(prediction - labels) == 0;
  BOOST_REQUIRE_EQUAL(b, true);
}

BOOST_AUTO_TEST_SUITE_END();
