/**
 * @file tests/activation_functions_test.cpp
 * @author Marcus Edel
 * @author Dhawal Arora
 *
 * Tests for the various activation functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/activation_functions/swish_function.hpp>
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>
#include <mlpack/methods/ann/activation_functions/mish_function.hpp>
#include <mlpack/methods/ann/activation_functions/lisht_function.hpp>
#include <mlpack/methods/ann/activation_functions/gelu_function.hpp>
#include <mlpack/methods/ann/activation_functions/elliot_function.hpp>
#include <mlpack/methods/ann/activation_functions/elish_function.hpp>
#include <mlpack/methods/ann/activation_functions/inverse_quadratic_function.hpp>
#include <mlpack/methods/ann/activation_functions/quadratic_function.hpp>
#include <mlpack/methods/ann/activation_functions/multi_quadratic_function.hpp>
#include <mlpack/methods/ann/activation_functions/spline_function.hpp>
#include <mlpack/methods/ann/activation_functions/poisson1_function.hpp>
#include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ann;

// Generate dataset for activation function tests.
const arma::colvec activationData("-2 3.2 4.5 -100.2 1 -1 2 0");

/**
 * Implementation of the activation function test.
 *
 * @param input Input data used for evaluating the activation function.
 * @param target Target data used to evaluate the activation.
 *
 * @tparam ActivationFunction Activation function used for the check.
 */
template<class ActivationFunction>
void CheckActivationCorrect(const arma::colvec input,
                            const arma::colvec target)
{
  // Test the activation function using a single value as input.
  for (size_t i = 0; i < target.n_elem; ++i)
  {
    REQUIRE(ActivationFunction::Fn(input.at(i)) ==
        Approx(target.at(i)).epsilon(1e-5));
  }

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  ActivationFunction::Fn(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the activation function derivative test.
 *
 * @param input Input data used for evaluating the activation function.
 * @param target Target data used to evaluate the activation.
 *
 * @tparam ActivationFunction Activation function used for the check.
 */
template<class ActivationFunction>
void CheckDerivativeCorrect(const arma::colvec input,
                            const arma::colvec target)
{
  // Test the calculation of the derivatives using a single value as input.
  for (size_t i = 0; i < target.n_elem; ++i)
  {
    REQUIRE(ActivationFunction::Deriv(input.at(i)) ==
        Approx(target.at(i)).epsilon(1e-5));
  }

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;
  ActivationFunction::Deriv(input, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
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
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    REQUIRE(ActivationFunction::Inv(ActivationFunction::Fn(input.at(i))) ==
        Approx(input.at(i)).epsilon(1e-5));
  }

  // Test the calculation of the inverse using the entire vector as input.
  arma::colvec activations;
  ActivationFunction::Fn(input, activations);
  ActivationFunction::Inv(activations, activations);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(input.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the HardTanH activation function test. The function is
 * implemented as a HardTanH Layer in hard_tanh.hpp
 *
 * @param input Input data used for evaluating the HardTanH activation function.
 * @param target Target data used to evaluate the HardTanH activation.
 */
void CheckHardTanHActivationCorrect(const arma::colvec input,
                                    const arma::colvec target)
{
  HardTanH<> htf;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  htf.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the HardTanH activation function derivative test. The
 * derivative is implemented as HardTanH Layer in hard_tanh.hpp
 *
 * @param input Input data used for evaluating the HardTanH activation
 * function.
 * @param target Target data used to evaluate the HardTanH activation.
 */
void CheckHardTanHDerivativeCorrect(const arma::colvec input,
                                    const arma::colvec target)
{
  HardTanH<> htf;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  htf.Backward(input, error, derivatives);

  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the LeakyReLU activation function test. The function is
 * implemented as LeakyReLU layer in the file leaky_relu.hpp
 *
 * @param input Input data used for evaluating the LeakyReLU activation
 * function.
 * @param target Target data used to evaluate the LeakyReLU activation.
 */
void CheckLeakyReLUActivationCorrect(const arma::colvec input,
                                     const arma::colvec target)
{
  LeakyReLU<> lrf;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  lrf.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the LeakyReLU activation function derivative test.
 * The derivative function is implemented as LeakyReLU layer in the file
 * leaky_relu_layer.hpp
 *
 * @param input Input data used for evaluating the LeakyReLU activation
 * function.
 * @param target Target data used to evaluate the LeakyReLU activation.
 */
void CheckLeakyReLUDerivativeCorrect(const arma::colvec input,
                                     const arma::colvec target)
{
  LeakyReLU<> lrf;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  lrf.Backward(input, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the ELU activation function test. The function is
 * implemented as ELU layer in the file elu.hpp
 *
 * @param input Input data used for evaluating the ELU activation function.
 * @param target Target data used to evaluate the ELU activation.
 */
void CheckELUActivationCorrect(const arma::colvec input,
                               const arma::colvec target)
{
  // Initialize ELU object with alpha = 1.0.
  ELU<> lrf(1.0);

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  lrf.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the ELU activation function derivative test. The function
 * is implemented as ELU layer in the file elu.hpp
 *
 * @param input Input data used for evaluating the ELU activation function.
 * @param target Target data used to evaluate the ELU activation.
 */
void CheckELUDerivativeCorrect(const arma::colvec input,
                               const arma::colvec target)
{
  // Initialize ELU object with alpha = 1.0.
  ELU<> lrf(1.0);

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives, activations;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  lrf.Forward(input, activations);
  lrf.Backward(activations, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the PReLU activation function test. The function
 * is implemented as PReLU layer in the file parametric_relu.hpp.
 *
 * @param input Input data used for evaluating the PReLU activation
 *   function.
 * @param target Target data used to evaluate the PReLU activation.
 */
void CheckPReLUActivationCorrect(const arma::colvec input,
                                 const arma::colvec target)
{
  PReLU<> prelu;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  prelu.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the PReLU activation function derivative test.
 * The function is implemented as PReLU layer in the file
 * parametric_relu.hpp
 *
 * @param input Input data used for evaluating the PReLU activation
 *   function.
 * @param target Target data used to evaluate the PReLU activation.
 */
void CheckPReLUDerivativeCorrect(const arma::colvec input,
                                 const arma::colvec target)
{
  PReLU<> prelu;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  prelu.Backward(input, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the PReLU activation function gradient test.
 * The function is implemented as PReLU layer in the file
 * parametric_relu.hpp
 *
 * @param input Input data used for evaluating the PReLU activation
 *   function.
 * @param target Target data used to evaluate the PReLU gradient.
 */
void CheckPReLUGradientCorrect(const arma::colvec input,
                               const arma::colvec target)
{
  PReLU<> prelu;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec gradient;

  // This error vector will be set to 1 to get the gradient.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  prelu.Gradient(input, error, gradient);
  REQUIRE(gradient.n_rows == 1);
  REQUIRE(gradient.n_cols == 1);
  REQUIRE(gradient(0) == Approx(target(0)).epsilon(1e-5));
}

/**
 * Implementation of the Hard Shrink activation function test. The function is
 * implemented as Hard Shrink layer in the file hardshrink.hpp
 *
 * @param input Input data used for evaluating the Hard Shrink activation function.
 * @param target Target data used to evaluate the Hard Shrink activation.
 */
void CheckHardShrinkActivationCorrect(const arma::colvec input,
                                      const arma::colvec target)
{
  HardShrink<> hardshrink;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  hardshrink.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the HardShrink activation function derivative test.
 * The derivative function is implemented as HardShrink layer in the file
 * hardshrink.hpp
 *
 * @param input Input data used for evaluating the HardShrink activation
 * function.
 * @param target Target data used to evaluate the HardShrink activation.
 */
void CheckHardShrinkDerivativeCorrect(const arma::colvec input,
                                      const arma::colvec target)
{
  HardShrink<> hardshrink;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  hardshrink.Backward(input, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the Soft Shrink activation function test. The function is
 * implemented as Soft Shrink layer in the file softshrink.hpp.
 *
 * @param input Input data used for evaluating the Soft Shrink activation
 * function.
 * @param target Target data used to evaluate the Soft Shrink activation.
 */
void CheckSoftShrinkActivationCorrect(const arma::colvec input,
                                      const arma::colvec target)
{
  SoftShrink<> softshrink;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  softshrink.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the Soft Shrink activation function derivative test.
 * The derivative function is implemented as Soft Shrink layer in the file
 * softshrink.hpp
 *
 * @param input Input data used for evaluating the Soft Shrink activation
 * function.
 * @param target Target data used to evaluate the Soft Shrink activation.
 */
void CheckSoftShrinkDerivativeCorrect(const arma::colvec input,
                                      const arma::colvec target)
{
  SoftShrink<> softshrink;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  softshrink.Backward(input, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Simple SELU activation test to check whether the mean and variance remain
 * invariant after passing normalized inputs through the function.
 */
TEST_CASE("SELUFunctionNormalizedTest", "[ActivationFunctionsTest]")
{
  arma::mat input = arma::randn<arma::mat>(1000, 1);

  arma::mat output;

  SELU selu;

  selu.Forward(input, output);

  REQUIRE(arma::as_scalar(arma::abs(arma::mean(input) -
      arma::mean(output))) <= 0.1);

  REQUIRE(arma::as_scalar(arma::abs(arma::var(input) -
      arma::var(output))) <= 0.1);
}

/**
 * Simple SELU activation test to check whether the mean and variance
 * vary significantly after passing unnormalized inputs through the function.
 */
TEST_CASE("SELUFunctionUnnormalizedTest", "[ActivationFunctionsTest]")
{
  const arma::colvec input("5.96402758 0.9966824 0.99975321 1 \
                            7.76159416 -0.76159416 0.96402758 8");

  arma::mat output;

  SELU selu;

  selu.Forward(input, output);

  REQUIRE(arma::as_scalar(arma::abs(arma::mean(input) -
      arma::mean(output))) >= 0.1);

  REQUIRE(arma::as_scalar(arma::abs(arma::var(input) -
      arma::var(output))) >= 0.1);
}

/**
 * Simple SELU derivative test to check whether the derivatives
 * produced by the activation function are correct.
 *
 */
TEST_CASE("SELUFunctionDerivativeTest", "[ActivationFunctionsTest]")
{
  arma::mat input = arma::ones<arma::mat>(1000, 1);

  arma::mat error = arma::ones<arma::mat>(input.n_elem, 1);

  arma::mat derivatives, activations;

  SELU selu;

  selu.Forward(input, activations);
  selu.Backward(activations, error, derivatives);

  REQUIRE(arma::as_scalar(arma::abs(arma::mean(derivatives) -
      selu.Lambda())) <= 10e-4);

  input.fill(-1);

  selu.Forward(input, activations);
  selu.Backward(activations, error, derivatives);

  REQUIRE(arma::as_scalar(arma::abs(arma::mean(derivatives) -
      selu.Lambda() * selu.Alpha() - arma::mean(activations))) <= 10e-4);
}

/**
 * Implementation of the CELU activation function test. The function is
 * implemented as CELU layer in the file celu.hpp.
 *
 * @param input Input data used for evaluating the CELU activation function.
 * @param target Target data used to evaluate the CELU activation.
 */
void CheckCELUActivationCorrect(const arma::colvec input,
                                const arma::colvec target)
{
  // Initialize CELU object with alpha = 1.0.
  CELU<> lrf(1.0);

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  lrf.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the CELU activation function derivative test. The function
 * is implemented as CELU layer in the file celu.hpp.
 *
 * @param input Input data used for evaluating the CELU activation function.
 * @param target Target data used to evaluate the CELU activation.
 */
void CheckCELUDerivativeCorrect(const arma::colvec input,
                                const arma::colvec target)
{
  // Initialize CELU object with alpha = 1.0.
  CELU<> lrf(1.0);

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives, activations;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  lrf.Forward(input, activations);
  lrf.Backward(activations, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the Softmin activation function test. The function is
 * implemented as Softmin layer in the file softmin.hpp.
 *
 * @param input Input data used for evaluating the Softmin activation function.
 * @param target Target data used to evaluate the Softmin activation.
 */
void CheckSoftminActivationCorrect(const arma::colvec input,
                                   const arma::colvec target)
{
  // Initialize Softmin object.
  Softmin<> softmin;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  softmin.Forward(input,activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}

/**
 * Implementation of the Softmin activation function derivative test.
 * The function is implemented as Softmin layer in the file softmin.hpp.
 *
 * @param input Input data used for evaluating the Softmin activation function.
 * @param target Target data used to evaluate the Softmin activation.
 */
void CheckSoftminDerivativeCorrect(const arma::colvec input,
                                   const arma::colvec target)
{
  // Initialize Softmin object.
  Softmin<> softmin;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives, activations;

  // This error vector will be set to [[1.0],[0.0],[1.0],[0.0]]
  // to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  error(1) = 0.0;
  error(3) = 0.0;
  softmin.Forward(input, activations);
  softmin.Backward(activations, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  } 

}

/**
 * Basic test of the tanh function.
 */
TEST_CASE("TanhFunctionTest", "[ActivationFunctionsTest]")
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
TEST_CASE("LogisticFunctionTest", "[ActivationFunctionsTest]")
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
TEST_CASE("SoftsignFunctionTest", "[ActivationFunctionsTest]")
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
TEST_CASE("IdentityFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredDerivatives = arma::ones<arma::colvec>(
      activationData.n_elem);

  CheckActivationCorrect<IdentityFunction>(activationData, activationData);
  CheckDerivativeCorrect<IdentityFunction>(activationData, desiredDerivatives);
}

/**
 * Basic test of the rectifier function.
 */
TEST_CASE("RectifierFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("0 3.2 4.5 0 1 0 2 0");

  const arma::colvec desiredDerivatives("0 1 1 0 1 0 1 0");

  CheckActivationCorrect<RectifierFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<RectifierFunction>(desiredActivations,
                                            desiredDerivatives);
}

/**
 * Basic test of the LeakyReLU function.
 */
TEST_CASE("LeakyReLUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-0.06 3.2 4.5 -3.006 \
                                         1 -0.03 2 0");

  const arma::colvec desiredDerivatives("0.03 1 1 0.03 \
                                         1 0.03 1 1");

  CheckLeakyReLUActivationCorrect(activationData, desiredActivations);
  CheckLeakyReLUDerivativeCorrect(desiredActivations, desiredDerivatives);
}

/**
 * Basic test of the HardTanH function.
 */
TEST_CASE("HardTanHFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-1 1 1 -1 \
                                         1 -1 1 0");

  const arma::colvec desiredDerivatives("0 0 0 0 \
                                         1 1 0 1");

  CheckHardTanHActivationCorrect(activationData, desiredActivations);
  CheckHardTanHDerivativeCorrect(activationData, desiredDerivatives);
}

/**
 * Basic test of the ELU function.
 */
TEST_CASE("ELUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-0.86466471 3.2 4.5 -1.0 \
                                         1 -0.63212055 2 0");

  const arma::colvec desiredDerivatives("0.13533529 1 1 0 \
                                         1 0.36787945 1 1");

  CheckELUActivationCorrect(activationData, desiredActivations);
  CheckELUDerivativeCorrect(activationData, desiredDerivatives);
}

/**
 * Basic test of the softplus function.
 */
TEST_CASE("SoftplusFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("-2 3.2 4.5 -100.2 1 -1 2 0 1000 10000");

  const arma::colvec desiredActivations("0.12692801 3.23995333 4.51104774 \
                                         0 1.31326168 0.31326168 2.12692801 \
                                         0.69314718 1000 10000");

  const arma::colvec desiredDerivatives("0.53168946 0.96231041 0.98913245 \
                                         0.5 0.78805844 0.57768119 0.89349302\
                                         0.66666666 1 1");

  CheckActivationCorrect<SoftplusFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<SoftplusFunction>(desiredActivations,
                                           desiredDerivatives);
  CheckInverseCorrect<SoftplusFunction>(desiredActivations);
}

/**
 * Basic test of the PReLU function.
 */
TEST_CASE("PReLUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-0.06 3.2 4.5 -3.006 \
                                         1 -0.03 2 0");

  const arma::colvec desiredDerivatives("0.03 1 1 0.03 \
                                         1 0.03 1 1");
  const arma::colvec desiredGradient("-103.2");

  CheckPReLUActivationCorrect(activationData, desiredActivations);
  CheckPReLUDerivativeCorrect(desiredActivations, desiredDerivatives);
  CheckPReLUGradientCorrect(activationData, desiredGradient);
}

/**
 * Basic test of the CReLU function.
 */
TEST_CASE("CReLUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("0 3.2 4.5 0 \
                                         1 0 2 0 2 0 0 \
                                         100.2 0 1 0 0");

  const arma::colvec desiredDerivatives("0 0 0 0 \
                                         0 0 0 0");
  CReLU<> crelu;
  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  crelu.Forward(activationData, activations);
  arma::colvec derivatives;
  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(desiredActivations.n_elem);
  crelu.Backward(desiredActivations, error, derivatives);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) ==
        Approx(desiredActivations.at(i)).epsilon(1e-5));
  }
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) ==
        Approx(desiredDerivatives.at(i)).epsilon(1e-5));
  }
}

/**
 * Basic test of the swish function.
 */
TEST_CASE("SwishFunctionTest", "[ActivationFunctionsTest]")
{
  // Hand-calculated values using Python interpreter.
  const arma::colvec desiredActivations("-0.238405 3.07466 4.45055 \
                                         -3.05183208657e-42 0.731058 -0.26894 \
                                         1.76159 0");

  const arma::colvec desiredDerivatives("0.3819171 1.0856295 1.039218 \
                                         0.5 0.83540367 0.3671335 1.073787\
                                         0.5");

  CheckActivationCorrect<SwishFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<SwishFunction>(desiredActivations,
                                        desiredDerivatives);
}

/**
 * Basic test of the hard sigmoid function.
 */
TEST_CASE("HardSigmoidFunctionTest", "[ActivationFunctionsTest]")
{
  // Hand-calculated values using Python interpreter.
  const arma::colvec desiredActivations("0.1 1 1 \
                                         0 0.7 0.3 \
                                         0.9 0.5");

  const arma::colvec desiredDerivatives("0.2 0.0 0.0 \
                                         0.0 0.2 0.2 0.2\
                                         0.2");

  CheckActivationCorrect<HardSigmoidFunction>(activationData,
                                              desiredActivations);
  CheckDerivativeCorrect<HardSigmoidFunction>(desiredActivations,
                                              desiredDerivatives);
}

/**
 * Basic test of the Mish function.
 */
TEST_CASE("MishFunctionTest", "[ActivationFunctionsTest]")
{
  // Calculated using tfa.activations.mish().
  // where tfa is tensorflow_addons.
  const arma::colvec desiredActivations("-0.25250152 3.1901977 \
                                         4.498914 -3.05183208e-42 0.86509836 \
                                         -0.30340138 1.943959 0");

  const arma::colvec desiredDerivatives("0.4382387  1.0159768849 \
                                         1.0019108 0.6 \
                                         1.0192586  0.40639898 \
                                         1.0725079  0.6");

  CheckActivationCorrect<MishFunction>(activationData,
                                       desiredActivations);
  CheckDerivativeCorrect<MishFunction>(desiredActivations,
                                       desiredDerivatives);
}

/**
 * Basic test of the LiSHT function.
 */
TEST_CASE("LiSHTFunctionTest", "[ActivationFunctionsTest]")
{
  // Calculated using tfa.activations.LiSHT().
  // where tfa is tensorflow_addons.
  const arma::colvec desiredActivations("1.928055 3.189384 \
                                         4.4988894 100.2 0.7615942 \
                                         0.7615942 1.9280552 0");

  const arma::colvec desiredDerivatives("1.1150033 1.0181904 \
                                         1.001978 1.0 \
                                         1.0896928 1.0896928 \
                                         1.1150033 0.0");

  CheckActivationCorrect<LiSHTFunction>(activationData,
                                        desiredActivations);
  CheckDerivativeCorrect<LiSHTFunction>(desiredActivations,
                                        desiredDerivatives);
}

/**
 * Basic test of the GELU function.
 */
TEST_CASE("GELUFunctionTest", "[ActivationFunctionsTest]")
{
  // Calculated using torch.nn.gelu().
  const arma::colvec desiredActivations("-0.0454023 3.1981304 \
                                         4.5 -0.0 0.84119199 \
                                         -0.158808 1.954597694 0.0");

  const arma::colvec desiredDerivatives("0.4637992 1.0065302 \
                                         1.0000293 0.5 1.03513446 \
                                         0.37435387 1.090984 0.5");

  CheckActivationCorrect<GELUFunction>(activationData,
                                       desiredActivations);
  CheckDerivativeCorrect<GELUFunction>(desiredActivations,
                                       desiredDerivatives);
}

/**
 * Basic test of the Hard Shrink function.
 */
TEST_CASE("HardShrinkFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-2 3.2 4.5 -100.2 1 -1 2 0");

  const arma::colvec desiredDerivatives("1 1 1 1 1 1 1 0");

  CheckHardShrinkActivationCorrect(activationData,
                                   desiredActivations);
  CheckHardShrinkDerivativeCorrect(desiredActivations,
                                   desiredDerivatives);
}

/**
 * Basic test of the Elliot function.
 */
TEST_CASE("ElliotFunctionTest", "[ActivationFunctionsTest]")
{
  // Calculated using PyTorch tensor.
  const arma::colvec desiredActivations("-0.66666667 0.76190476 0.81818182 \
                                         -0.99011858 0.5 -0.5 \
                                          0.66666667 0.0 ");

  const arma::colvec desiredDerivatives("0.36 0.32213294 0.3025 \
                                         0.25248879 0.44444444 \
                                         0.44444444 0.36 1.0 ");

  CheckActivationCorrect<ElliotFunction>(activationData,
                                         desiredActivations);
  CheckDerivativeCorrect<ElliotFunction>(desiredActivations,
                                         desiredDerivatives);
}

/**
 * Basic test of the EliSH function.
 */
TEST_CASE("ElishFunctionTest", "[ActivationFunctionsTest]")
{
  // Manually-calculated using python-numpy module.
  const arma::colvec desiredActivations("-0.10307056 3.0746696 4.4505587 \
                                         -3.0457406e-44 0.731058578 \
                                         -0.1700034 1.76159415 0.0 ");

  const arma::colvec desiredDerivatives("0.4033889 1.0856292 \
                                         1.03921798 0.5 0.83540389 \
                                         0.34725726 1.07378804 0.5");

  CheckActivationCorrect<ElishFunction>(activationData,
                                        desiredActivations);
  CheckDerivativeCorrect<ElishFunction>(desiredActivations,
                                        desiredDerivatives);
}

/**
 * Basic test of the Soft Shrink function.
 */
TEST_CASE("SoftShrinkFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-1.5 2.7 4 -99.7 0.5 -0.5 1.5 0");

  const arma::colvec desiredDerivatives("1 1 1 1 1 1 1 0");

  CheckSoftShrinkActivationCorrect(activationData,
                                   desiredActivations);
  CheckSoftShrinkDerivativeCorrect(desiredActivations,
                                   desiredDerivatives);
}

/**
 * Basic test of the CELU activation function.
 */
TEST_CASE("CELUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-0.86466472 3.2 4.5 \
                                         -1 1 -0.63212056 2 0");

  const arma::colvec desiredDerivatives("0.42119275 1 1 \
                                         0.36787944 1 \
                                         0.5314636 1 1");

  CheckCELUActivationCorrect(activationData, desiredActivations);
  CheckCELUDerivativeCorrect(desiredActivations, desiredDerivatives);
}

/**
 * Basic test of the inverse quadratic function.
 */
TEST_CASE("InverseQuadraticFunctionTest", "[ActivationFunctionsTest]")
{
  // Hand-calculated values.
  const arma::colvec desiredActivations("0.2 0.088968 0.0470588 \
                                         9.95913e-05 0.5 0.5 \
                                         0.2 1");

  const arma::colvec desiredDerivatives("-0.369822 -0.175152 -0.0937021 \
                                         -0.000199183 -0.64 -0.64 -0.369822\
                                         -0.5");

  CheckActivationCorrect<InvQuadFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<InvQuadFunction>(desiredActivations,
                                          desiredDerivatives);
}

/**
 * Basic test of the quadratic function.
 */
TEST_CASE("QuadraticFunctionTest", "[ActivationFunctionsTest]")
{
  // Hand-calculated values.
  const arma::colvec desiredActivations("4 10.24 20.25 \
                                         10040 1 1 \
                                         4 0");

  const arma::colvec desiredDerivatives("8 20.48 40.50 \
                                         20080 2 2 \
                                         8 0");

  CheckActivationCorrect<QuadraticFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<QuadraticFunction>(desiredActivations,
                                            desiredDerivatives);
}

/**
 * Basic test of the Spline function.
 */
TEST_CASE("SplineFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData1("2 3.2 4.5 100.2 1 1 2 0");

  // Hand-calculated values.
  const arma::colvec desiredActivations("4.39445 14.6953 34.5211 \
                                         46355.9 0.693147 0.693147 \
                                         4.39445 0");

  const arma::colvec desiredDerivatives("18.3923 94.6819 280.03866 \
                                         1042462.1078 1.0137702 1.0137702 \
                                         18.3923 0");

  CheckActivationCorrect<SplineFunction>(activationData1, desiredActivations);
  CheckDerivativeCorrect<SplineFunction>(desiredActivations,
                                         desiredDerivatives);
}

/**
 * Basic test of the multi quadratic function.
 */
TEST_CASE("MultiquadFunctionTest", "[ActivationFunctionsTest]")
{
  // Hand-calculated values.
  const arma::colvec desiredActivations("2.23607 3.35261 4.60977 \
                                         100.205 1.41421 1.41421 \
                                         2.23607 1");

  const arma::colvec desiredDerivatives("0.912871 0.95828 0.97727 \
                                         0.99995 0.816496 0.816496 \
                                         0.912871 0.707107");

  CheckActivationCorrect<MultiQuadFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<MultiQuadFunction>(desiredActivations,
                                            desiredDerivatives);
}


/**
 * Basic test of the Poisson one function.
 */
TEST_CASE("Poisson1FunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData1("-2 3.2 4.5 5 1 -1 2 0");

  // Hand-calculated values.
  const arma::colvec desiredActivations("-22.1672 0.0896768 0.0388815 \
                                         0.0269518 0 -5.43656 \
                                         0.135335 -1");

  const arma::colvec desiredDerivatives("1.02404e+11 1.74647 1.88633 \
                                         1.92058 2 1707.81 \
                                         1.62864 8.15485");

  CheckActivationCorrect<Poisson1Function>(activationData1, desiredActivations);
  CheckDerivativeCorrect<Poisson1Function>(desiredActivations,
                                           desiredDerivatives);
}

/**
 * Basic test of the Gaussian activation function.
 */
TEST_CASE("GaussianFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("0.018315639 0.000035713 \
                                         1.6052280551856116e-09 \
                                         0 0.367879441 0.367879441 \
                                         0.018315639 1");

  const arma::colvec desiredDerivatives("-0.036618991635992616 \
                                         -0.0000714259999 \
                                         -0.0000000032104561 \
                                         0 -0.6426287436 \
                                         -0.642628743680 \
                                         -0.03661899163 \
                                         -0.73575888234");

  CheckActivationCorrect<GaussianFunction>(activationData,
                                           desiredActivations);
  CheckDerivativeCorrect<GaussianFunction>(desiredActivations,
                                           desiredDerivatives);
}

/**
 * Basic test of the Softmin function.
 */
TEST_CASE("SoftminFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("4.2 2.4 7.0 6.4");

  // Hand-calculated Values.
  const arma::colvec desiredActivations("0.1384799751 0.8377550303 \
                                         0.008420976 0.0153440186");

  const arma::colvec desiredDerivatives("0.1181371351 -0.12306701070 \
                                         0.0071839266 -0.0022540509");

  CheckSoftminActivationCorrect(activationData,
                                desiredActivations);
  CheckSoftminDerivativeCorrect(activationData,
                                desiredDerivatives);
}
