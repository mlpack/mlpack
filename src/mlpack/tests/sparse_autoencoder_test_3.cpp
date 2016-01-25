/**
 * @file sparse_autoencoder_test.cpp
 * @author Tham Ngap Wei
 *
 * Test the SparseAutoencoder class.
 */

#define BOOST_TEST_MODULE SparseAutoencoder

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_input_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_output_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_bias_layer.hpp>

#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/performance_functions/sparse_function.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(SparseAutoencoderTest);

using Network = std::tuple<SparseInputLayer<>, SparseBiasLayer<>,
                           BaseLayer<LogisticFunction>, SparseOutputLayer<>, SparseBiasLayer<>,
                           BaseLayer<LogisticFunction> >;

using FFNet = FFN<Network, OneHotLayer, SparseErrorFunction<> >;

Network create_network(size_t visibleSize, size_t hiddenSize,
                       size_t sampleSize, double lambda = 0.0001,
                       double beta = 3, double rho = 0.01)
{
  const double range = std::sqrt(6) / std::sqrt(visibleSize + hiddenSize + 1);

  SparseInputLayer<> hiddenLayer(visibleSize, hiddenSize, {-range, range},
                                 lambda);
  SparseBiasLayer<> hiddenBiasLayer(hiddenSize, sampleSize);
  BaseLayer<LogisticFunction> hiddenBaseLayer;

  SparseOutputLayer<> outputLayer(hiddenSize, visibleSize, {-range, range},
                                  lambda, beta, rho);
  SparseBiasLayer<> outputBiasLayer(visibleSize, sampleSize);
  BaseLayer<LogisticFunction> outputBaseLayer;

  return std::make_tuple(std::move(hiddenLayer), std::move(hiddenBiasLayer),
                         std::move(hiddenBaseLayer), std::move(outputLayer),
                         std::move(outputBiasLayer), std::move(outputBaseLayer));
}

FFNet create_ffn(size_t visibleSize, size_t hiddenSize,
                 size_t sampleSize, double lambda = 0.0001,
                 double beta = 3, double rho = 0.01)
{
  auto network = create_network(visibleSize, hiddenSize,
                                sampleSize, lambda,
                                beta, rho);
  FFNet ffn(std::move(network), OneHotLayer(),
            SparseErrorFunction<>(lambda,beta,rho));

  return ffn;
}

template<typename Net, typename UnaryFunc>
void initWeights(Net &net, UnaryFunc func)
{
  std::get<0>(net).Weights() = func(std::get<0>(net).Weights());
  std::get<1>(net).Weights() = func(std::get<1>(net).Weights());
  std::get<3>(net).Weights() = func(std::get<3>(net).Weights());
  std::get<4>(net).Weights() = func(std::get<4>(net).Weights());
}

template<typename Net>
void initWeights(Net &net, arma::mat const &parameters)
{
  const size_t l1 = (parameters.n_rows - 1) / 2;
  const size_t l2 = parameters.n_cols - 1;
  const size_t l3 = 2 * l1;
  std::get<0>(net).Weights() = parameters.submat(0, 0, l1-1, l2-1);
  std::get<1>(net).Weights() = parameters.submat(0, l2, l1-1, l2);
  std::get<3>(net).Weights() = parameters.submat(l1, 0, l3-1, l2-1).t();
  std::get<4>(net).Weights() = parameters.submat(l3, 0, l3, l2-1).t();
}

arma::mat initGradient(arma::mat const &input,
                       arma::mat const &parameters,
                       FFNet &ffn)
{
  initWeights(ffn.Network(), parameters);
  arma::mat error;
  ffn.FeedForward(input, input, error);
  ffn.FeedBackward(arma::mat(), error);

  arma::mat gradient = arma::zeros(parameters.n_rows, parameters.n_cols);
  const size_t l1 = (parameters.n_rows - 1) / 2;
  const size_t l2 = parameters.n_cols - 1;
  const size_t l3 = 2 * l1;
  auto const &net = ffn.Network();
  gradient.submat(0, 0, l1-1, l2-1) = std::get<0>(net).Gradient();
  gradient.submat(0, l2, l1-1, l2) = std::get<1>(net).Gradient();
  gradient.submat(l1, 0, l3-1, l2-1) = std::get<3>(net).Gradient().t();
  gradient.submat(l3, 0, l3, l2-1) = std::get<4>(net).Gradient().t();

  return gradient;
}

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionEvaluate)
{
  const size_t vSize = 5;
  const size_t hSize = 3;  

  // Simple fake dataset.
  arma::mat data1("0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5;"
                  "0.1 0.2 0.3 0.4 0.5");

  // Create SparseAutoencoder. Regularization and KL divergence terms
  // ignored.
  auto ffn1 = create_ffn(vSize, hSize, 5, 0, 0);
  initWeights(ffn1.Network(), [](arma::mat &v){
    return v.ones();
  });
  arma::mat error;
  ffn1.FeedForward(data1, data1, error);

  auto ffn2 = create_ffn(vSize, hSize, 5, 0, 0);
  initWeights(ffn2.Network(), [](arma::mat &v){
    return v.zeros();
  });
  ffn2.FeedForward(data1, data1, error);

  auto ffn3 = create_ffn(vSize, hSize, 5, 0, 0);
  initWeights(ffn3.Network(), [](arma::mat &v){
    return -v.ones();
  });
  ffn3.FeedForward(data1, data1, error);

  // Test using first dataset. Values were calculated using Octave.
  BOOST_REQUIRE_CLOSE(ffn1.Error(), 1.190472606540, 1e-5);
  BOOST_REQUIRE_CLOSE(ffn2.Error(), 0.150000000000, 1e-5);
  BOOST_REQUIRE_CLOSE(ffn3.Error(), 0.048800332266, 1e-5);

  arma::mat const data2 = data1.t();
  auto ffn4 = create_ffn(vSize, hSize, 5, 0, 0);
  initWeights(ffn4.Network(), [](arma::mat &v){
    return v.ones();
  });
  ffn4.FeedForward(data2, data2, error);

  auto ffn5 = create_ffn(vSize, hSize, 5, 0, 0);
  initWeights(ffn5.Network(), [](arma::mat &v){
    return v.zeros();
  });
  ffn5.FeedForward(data2, data2, error);

  auto ffn6 = create_ffn(vSize, hSize, 5, 0, 0);
  initWeights(ffn6.Network(), [](arma::mat &v){
    return -v.ones();
  });
  ffn6.FeedForward(data2, data2, error);

  // Test using second dataset. Values were calculated using Octave.
  BOOST_REQUIRE_CLOSE(ffn4.Error(), 1.197585812647, 1e-5);
  BOOST_REQUIRE_CLOSE(ffn5.Error(), 0.150000000000, 1e-5);
  BOOST_REQUIRE_CLOSE(ffn6.Error(), 0.063466617408, 1e-5);
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
  arma::mat data1;
  data1.randu(vSize, points);

  // Run a number of trials.
  for(size_t i = 0; i < trials; i++)
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
                                     data1.col(j) + parameters.submat(0, l2, l1 - 1, l2))));
      outputLayer = 1.0 /
                    (1 + arma::exp(-(parameters.submat(l1, 0, l3 - 1,l2 - 1).t()
                                     * hiddenLayer + parameters.submat(l3, 0, l3, l2 - 1).t())));
      diff = outputLayer - data1.col(j);

      reconstructionError += 0.5 * arma::sum(arma::sum(diff % diff));
    }
    reconstructionError /= points;

    // Compare with the value returned by the function.
    auto ffn = create_ffn(vSize, hSize, points, 0, 0);
    auto &net = ffn.Network();
    initWeights(net, parameters);
    arma::mat error;
    ffn.FeedForward(data1, data1, error);
    BOOST_REQUIRE_CLOSE(ffn.Error(), reconstructionError, 1e-5);
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

    // 3 objects for comparing regularization costs.
    auto safNoReg = create_ffn(vSize, hSize, points, 0, 0);
    initWeights(safNoReg.Network(), parameters);
    arma::mat error;
    safNoReg.FeedForward(data, data, error);

    auto safSmallReg = create_ffn(vSize, hSize, points, 0.5, 0);
    initWeights(safSmallReg.Network(), parameters);
    safSmallReg.FeedForward(data, data, error);

    auto safBigReg = create_ffn(vSize, hSize, points, 20, 0);
    initWeights(safBigReg.Network(), parameters);
    safBigReg.FeedForward(data, data, error);

    BOOST_REQUIRE_CLOSE(safNoReg.Error() + smallRegTerm,
                        safSmallReg.Error(), 1e-5);
    BOOST_REQUIRE_CLOSE(safNoReg.Error() + bigRegTerm,
                        safBigReg.Error(), 1e-5);
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
    rhoCap /= static_cast<double>(points);

    // Calculate divergence terms.
    const double smallDivTerm = 5 * arma::accu(rho * arma::log(rho / rhoCap) +
                                               (1 - rho) * arma::log((1 - rho) / (1 - rhoCap)));
    const double bigDivTerm = 20 * arma::accu(rho * arma::log(rho / rhoCap) +
                                              (1 - rho) * arma::log((1 - rho) / (1 - rhoCap)));

    // 3 objects for comparing divergence costs.
    auto safNoDiv = create_ffn(vSize, hSize, points, 0, 0, rho);
    initWeights(safNoDiv.Network(), parameters);
    arma::mat error;
    safNoDiv.FeedForward(data, data, error);

    auto safSmallDiv = create_ffn(vSize, hSize, points, 0, 5, rho);
    initWeights(safSmallDiv.Network(), parameters);
    safSmallDiv.FeedForward(data, data, error);

    auto safBigDiv = create_ffn(vSize, hSize, points, 0, 20, rho);
    initWeights(safBigDiv.Network(), parameters);
    safBigDiv.FeedForward(data, data, error);

    BOOST_REQUIRE_CLOSE(safNoDiv.Error() + smallDivTerm,
                        safSmallDiv.Error(), 1e-5);
    BOOST_REQUIRE_CLOSE(safNoDiv.Error() + bigDivTerm,
                        safBigDiv.Error(), 1e-5);
  }
}//*/

BOOST_AUTO_TEST_CASE(SparseAutoencoderFunctionGradient)
{
  const arma::mat trainData("0.0013   0.5850   0.8228   0.7105;"
                            "0.1933   0.3503   0.1741   0.3040");
  const size_t visibleSize = trainData.n_rows;
  const size_t hiddenSize = trainData.n_rows / 2;

  arma::mat const parameters( "-1.2029  -0.8175        0;"
                              " 0.0776  -0.1205        0;"
                              "      0        0        0");

  auto ffn = create_ffn(visibleSize, hiddenSize, 4);
  initGradient(trainData, parameters, ffn);
  auto &net = ffn.Network();

  arma::mat const w1 = std::get<0>(net).Gradient();
  arma::mat const b1 = std::get<1>(net).Gradient();
  arma::mat const w2 = std::get<3>(net).Gradient();
  arma::mat const b2 = std::get<4>(net).Gradient();

  //these values come from original implementation
  //because ffn do not provide ease to use Evalulate
  //api to do the gradient checking, I prefer golden
  //model(FeedForward api will alter inner value,
  //everytime you call it, the error value will change)
  BOOST_REQUIRE_CLOSE(w1(0, 0), 0.41743157405, 1e-5);
  BOOST_REQUIRE_CLOSE(w1(0, 1), 0.21509458293, 1e-5);
  BOOST_REQUIRE_CLOSE(b1(0, 0), 0.85303876677, 1e-5);
  BOOST_REQUIRE_CLOSE(w2(0, 0), 0.00520531433, 1e-5);
  BOOST_REQUIRE_CLOSE(w2(1, 0), 0.01858690990, 1e-5);
  BOOST_REQUIRE_CLOSE(b2(0, 0), -0.00599756012, 1e-5);
  BOOST_REQUIRE_CLOSE(b2(1, 0), 0.05881613659, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
