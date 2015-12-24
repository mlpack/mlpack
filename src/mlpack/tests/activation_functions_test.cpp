/**
 * @file activation_functions_test.cpp
 * @author Marcus Edel
 *
 * Tests for the various activation functions.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>

#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ActivationFunctionsTest);

// Be careful!  When writing new tests, always get the boolean value and store
// it in a temporary, because the Boost unit test macros do weird things and
// will cause bizarre problems.

// Generate dataset for activation function tests.
const arma::colvec activationData("-2 3.2 4.5 -100.2 1 -1 2 0");

/*
 * Implementation of the activation function test.
 *
 * @param input Input data used for evaluating the activation function.
 * @param target Target data used to evaluate the activation.
 *
 * @tparam ActivationFunction Activation function used for the check.
 */
template<class ActivationFunction>
void CheckActivationCorrect(const arma::colvec input, const arma::colvec target)
{
  // Test the activation function using a single value as input.
  for (size_t i = 0; i < target.n_elem; i++)
  {
    BOOST_REQUIRE_CLOSE(ActivationFunction::fn(input.at(i)),
        target.at(i), 1e-3);
  }

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  ActivationFunction::fn(input, activations);
  for (size_t i = 0; i < activations.n_elem; i++)
  {
    BOOST_REQUIRE_CLOSE(activations.at(i), target.at(i), 1e-3);
  }
}

/*
 * Implementation of the activation function derivative test.
 *
 * @param input Input data used for evaluating the activation function.
 * @param target Target data used to evaluate the activation.
 *
 * @tparam ActivationFunction Activation function used for the check.
 */
template<class ActivationFunction>
void CheckDerivativeCorrect(const arma::colvec input, const arma::colvec target)
{
  // Test the calculation of the derivatives using a single value as input.
  for (size_t i = 0; i < target.n_elem; i++)
  {
    BOOST_REQUIRE_CLOSE(ActivationFunction::deriv(input.at(i)),
        target.at(i), 1e-3);
  }

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;
  ActivationFunction::deriv(input, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; i++)
  {
    BOOST_REQUIRE_CLOSE(derivatives.at(i), target.at(i), 1e-3);
  }
}

/*
 * Implementation of the activation function inverse test.
 *
 * @param input Input data used for evaluating the activation function.
 * @param target Target data used to evaluate the activation.
 *
 * @tparam ActivationFunction Activation function used for the check.
 */
template<class ActivationFunction>
void CheckInverseCorrect(const arma::colvec input)
{
    // Test the calculation of the inverse using a single value as input.
  for (size_t i = 0; i < input.n_elem; i++)
  {
    BOOST_REQUIRE_CLOSE(ActivationFunction::inv(ActivationFunction::fn(
        input.at(i))), input.at(i), 1e-3);
  }

  // Test the calculation of the inverse using the entire vector as input.
  arma::colvec activations;
  ActivationFunction::fn(input, activations);
  ActivationFunction::inv(activations, activations);

  for (size_t i = 0; i < input.n_elem; i++)
  {
    BOOST_REQUIRE_CLOSE(activations.at(i), input.at(i), 1e-3);
  }
}

/**
 * Basic test of the tanh function.
 */
BOOST_AUTO_TEST_CASE(TanhFunctionTest)
{
  const arma::colvec desiredActivations("-0.96402758 0.9966824 0.99975321 -1 \
                                         0.76159416 -0.76159416 0.96402758 0");

  const arma::colvec desiredDerivatives("0.07065082 0.00662419 0.00049352 0 \
                                         0.41997434 0.41997434 0.07065082 1");

  CheckActivationCorrect<TanhFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<TanhFunction>(desiredActivations, desiredDerivatives);
  CheckInverseCorrect<TanhFunction>(desiredActivations);
}

/**
 * Basic test of the logistic function.
 */
BOOST_AUTO_TEST_CASE(LogisticFunctionTest)
{
  const arma::colvec desiredActivations("1.19202922e-01 9.60834277e-01 \
                                         9.89013057e-01 3.04574e-44 \
                                         7.31058579e-01 2.68941421e-01 \
                                         8.80797078e-01 0.5");

  const arma::colvec desiredDerivatives("0.10499359 0.03763177 0.01086623 \
                                         3.04574e-44 0.19661193 0.19661193 \
                                         0.10499359 0.25");

  CheckActivationCorrect<LogisticFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<LogisticFunction>(desiredActivations,
      desiredDerivatives);
  CheckInverseCorrect<LogisticFunction>(activationData);
}

/**
 * Basic test of the softsign function.
 */
BOOST_AUTO_TEST_CASE(SoftsignFunctionTest)
{
  const arma::colvec desiredActivations("-0.66666667 0.76190476 0.81818182 \
                                         -0.99011858 0.5 -0.5 0.66666667 0");

  const arma::colvec desiredDerivatives("0.11111111 0.05668934 0.03305785 \
                                         9.7642e-05 0.25 0.25 0.11111111 1");

  CheckActivationCorrect<SoftsignFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<SoftsignFunction>(desiredActivations,
      desiredDerivatives);
  CheckInverseCorrect<SoftsignFunction>(desiredActivations);
}

/**
 * Basic test of the identity function.
 */
BOOST_AUTO_TEST_CASE(IdentityFunctionTest)
{
  const arma::colvec desiredDerivatives = arma::ones<arma::colvec>(
      activationData.n_elem);

  CheckActivationCorrect<IdentityFunction>(activationData, activationData);
  CheckDerivativeCorrect<IdentityFunction>(activationData, desiredDerivatives);
}

/**
 * Basic test of the rectifier function.
 */
BOOST_AUTO_TEST_CASE(RectifierFunctionTest)
{
  const arma::colvec desiredActivations("0 3.2 4.5 0 1 0 2 0");

  const arma::colvec desiredDerivatives("0 1 1 0 1 0 1 0");

  CheckActivationCorrect<RectifierFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<RectifierFunction>(desiredActivations,
      desiredDerivatives);
}

/*
 * Implementation of the numerical gradient checking.
 *
 * @param input Input data used for evaluating the network.
 * @param target Target data used to calculate the network error.
 * @param perturbation Constant perturbation value.
 * @param threshold Threshold used as bounding check.
 *
 * @tparam ActivationFunction Activation function used for the gradient check.
 */
template<class ActivationFunction>
void CheckGradientNumericallyCorrect(const arma::mat input,
                                     const arma::mat target,
                                     const double perturbation,
                                     const double threshold)
{
  // Specify the structure of the feed forward neural network.
  RandomInitialization randInit(-0.5, 0.5);
  arma::mat error;

  // Number of hidden layer units.
  const size_t hiddenLayerSize = 4;

  LinearLayer<mlpack::ann::RMSPROP, RandomInitialization> linearLayer0(
        input.n_rows, hiddenLayerSize, randInit);
  BiasLayer<> biasLayer0(hiddenLayerSize);
  BaseLayer<ActivationFunction> baseLayer0;

  LinearLayer<mlpack::ann::RMSPROP, RandomInitialization> linearLayer1(
         hiddenLayerSize, hiddenLayerSize, randInit);
  BiasLayer<> biasLayer1(hiddenLayerSize);
  BaseLayer<ActivationFunction> baseLayer1;

  LinearLayer<mlpack::ann::RMSPROP, RandomInitialization> linearLayer2(
         hiddenLayerSize, target.n_rows, randInit);
  BiasLayer<> biasLayer2(target.n_rows);
  BaseLayer<ActivationFunction> baseLayer2;

  BinaryClassificationLayer classOutputLayer;

  auto modules = std::tie(linearLayer0, biasLayer0, baseLayer0,
                          linearLayer1, biasLayer1, baseLayer1,
                          linearLayer2, biasLayer2, baseLayer2);

  FFN<decltype(modules), decltype(classOutputLayer), MeanSquaredErrorFunction>
      net(modules, classOutputLayer);

  // Initialize the feed forward neural network.
  net.FeedForward(input, target, error);
  net.FeedBackward(input, error);

  std::vector<std::reference_wrapper<decltype(linearLayer0)> > layer {
         linearLayer0, linearLayer1, linearLayer2 };

  std::vector<arma::mat> gradient {linearLayer0.Gradient(),
                                   linearLayer1.Gradient(),
                                   linearLayer2.Gradient()};

  double weight, mLoss, pLoss, dW, e;

  for (size_t l = 0; l < layer.size(); ++l)
  {
    for (size_t i = 0; i < layer[l].get().Weights().n_rows; ++i)
    {
      for (size_t j = 0; j < layer[l].get().Weights().n_cols; ++j)
      {
        // Store original weight.
        weight = layer[l].get().Weights()(i, j);

        // Add negative perturbation and compute error.
        layer[l].get().Weights().at(i, j) -= perturbation;
        net.FeedForward(input, target, error);
        mLoss = arma::as_scalar(0.5 * arma::sum(arma::pow(error, 2)));

        // Add positive perturbation and compute error.
        layer[l].get().Weights().at(i, j) += (2 * perturbation);
        net.FeedForward(input, target, error);
        pLoss = arma::as_scalar(0.5 * arma::sum(arma::pow(error, 2)));

        // Compute symmetric difference.
        dW = (pLoss - mLoss) / (2 * perturbation);
        e = std::abs(dW - gradient[l].at(i, j));

        bool b = e < threshold;
        BOOST_REQUIRE_EQUAL(b, 1);

        // Restore original weight.
        layer[l].get().Weights().at(i, j) = weight;
      }
    }
  }
}

/**
 * The following test implements numerical gradient checking. It computes the
 * numerical gradient, a numerical approximation of the partial derivative of J
 * with respect to the i-th input argument, evaluated at g. The numerical
 * gradient should be approximately the partial derivative of J with respect to
 * g(i).
 *
 * Given a function g(\theta) that is supposedly computing:
 *
 * @f[
 * \frac{\partial}{\partial \theta} J(\theta)
 * @f]
 *
 * we can now numerically verify its correctness by checking:
 *
 * @f[
 * g(\theta) \approx \frac{J(\theta + eps) - J(\theta - eps)}{2 * eps}
 * @f]
 */
BOOST_AUTO_TEST_CASE(GradientNumericallyCorrect)
{
  // Initialize dataset.
  const arma::colvec input = arma::randu<arma::colvec>(10);
  const arma::colvec target("0 1;");

  // Perturbation and threshold constant.
  const double perturbation = 1e-6;
  const double threshold = 1e-5;

  CheckGradientNumericallyCorrect<LogisticFunction>(input, target,
      perturbation, threshold);

  CheckGradientNumericallyCorrect<IdentityFunction>(input, target,
      perturbation, threshold);

  CheckGradientNumericallyCorrect<RectifierFunction>(input, target,
      perturbation, threshold);

  CheckGradientNumericallyCorrect<SoftsignFunction>(input, target,
      perturbation, threshold);

  CheckGradientNumericallyCorrect<TanhFunction>(input, target,
      perturbation, threshold);
}

BOOST_AUTO_TEST_SUITE_END();
