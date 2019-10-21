
#include <ensmallen.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(CallbackTest);

/**
 * Test a FFN model with PrintLoss callback.
 */
BOOST_AUTO_TEST_CASE(FFNCallbackTest)
{
  arma::mat data;
  arma::mat labels;

  data::Load("lab1.csv", data, true);
  data::Load("lab3.csv", labels, true);

  FFN<MeanSquaredError<>, RandomInitialization> model;

  model.Add<Linear<>>(1, 2);
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>(2, 1);
  model.Add<SigmoidLayer<>>();

  std::stringstream stream;
  model.Train(data, labels, ens::PrintLoss(stream));

  BOOST_REQUIRE_GT(stream.str().length(), 0);
}

/**
 * Test a FFN model with PrintLoss callback and optimizer parameter.
 */
BOOST_AUTO_TEST_CASE(FFNWithOptimizerCallbackTest)
{
  arma::mat data;
  arma::mat labels;

  data::Load("lab1.csv", data, true);
  data::Load("lab3.csv", labels, true);

  FFN<MeanSquaredError<>, RandomInitialization> model;

  model.Add<Linear<>>(1, 2);
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>(2, 1);
  model.Add<SigmoidLayer<>>();

  std::stringstream stream;
  ens::StandardSGD opt(0.1, 1, 5);
  model.Train(data, labels, opt, ens::PrintLoss(stream));

  BOOST_REQUIRE_GT(stream.str().length(), 0);
}

/**
 * Test a RNN model with PrintLoss callback.
 */
BOOST_AUTO_TEST_CASE(RNNCallbackTest)
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> model(
      rho, false, NegativeLogLikelihood<>(), init);
  model.Add<IdentityLayer<> >();
  model.Add<Linear<> >(1, 10);

  // Use LSTM layer with rho.
  model.Add<LSTM<> >(10, 3, rho);
  model.Add<LogSoftMax<> >();

  std::stringstream stream;
  model.Train(input, target, ens::PrintLoss(stream));

  BOOST_REQUIRE_GT(stream.str().length(), 0);
}

/**
 * Test a RNN model with PrintLoss callback and optimizer parameter.
 */
BOOST_AUTO_TEST_CASE(RNNWithOptimizerCallbackTest)
{
  const size_t rho = 5;
  arma::cube input = arma::randu(1, 1, 5);
  arma::cube target = arma::ones(1, 1, 5);
  RandomInitialization init(0.5, 0.5);

  // Create model with user defined rho parameter.
  RNN<NegativeLogLikelihood<>, RandomInitialization> model(
      rho, false, NegativeLogLikelihood<>(), init);
  model.Add<IdentityLayer<> >();
  model.Add<Linear<> >(1, 10);

  // Use LSTM layer with rho.
  model.Add<LSTM<> >(10, 3, rho);
  model.Add<LogSoftMax<> >();

  std::stringstream stream;
  ens::StandardSGD opt(0.1, 1, 5);
  model.Train(input, target, opt, ens::PrintLoss(stream));

  BOOST_REQUIRE_GT(stream.str().length(), 0);
}

/**
 *  Test Logistic regression implementation with Printloss callback.
 */

BOOST_AUTO_TEST_CASE(LRWithOptimizerCallback)
{
    arma::mat data("1 2 3;"
                   "1 2 3");
    arma::Row<size_t> responses("1 1 0");
    ens::StandardSGD sgd(0.1, 1, 5);
    LogisticRegression<> logisticRegression(data, responses, sgd, 0.001);
    std::stringstream stream;
    logisticRegression.Train<ens::StandardSGD>(
            data,
            responses,
            sgd,
            ens::PrintLoss(stream));
    BOOST_REQUIRE_GT(stream.str().length(), 0);
}

BOOST_AUTO_TEST_SUITE_END();
