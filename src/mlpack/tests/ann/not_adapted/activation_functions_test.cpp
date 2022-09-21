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

#include <mlpack/methods/ann/layer/layer_types.hpp>
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
#include <mlpack/methods/ann/activation_functions/hard_swish_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_exponential_function.hpp>
#include <mlpack/methods/ann/activation_functions/silu_function.hpp>

#include "../catch.hpp"

using namespace mlpack;

/**
 * Implementation of the HardTanH activation function test. The function is
 * implemented as a HardTanH Layer in hard_tanh.hpp
 *
 * @param input Input data used for evaluating the HardTanH activation function.
 * @param target Target data used to evaluate the HardTanH activation.
 *
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
}*/

/**
 * Implementation of the HardTanH activation function derivative test. The
 * derivative is implemented as HardTanH Layer in hard_tanh.hpp
 *
 * @param input Input data used for evaluating the HardTanH activation
 * function.
 * @param target Target data used to evaluate the HardTanH activation.
 *
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
}*/

/**
 * Implementation of the PReLU activation function test. The function
 * is implemented as PReLU layer in the file parametric_relu.hpp.
 *
 * @param input Input data used for evaluating the PReLU activation
 *   function.
 * @param target Target data used to evaluate the PReLU activation.
 *
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
}*/

/**
 * Implementation of the PReLU activation function derivative test.
 * The function is implemented as PReLU layer in the file
 * parametric_relu.hpp
 *
 * @param input Input data used for evaluating the PReLU activation
 *   function.
 * @param target Target data used to evaluate the PReLU activation.
 *
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
}*/

/**
 * Implementation of the PReLU activation function gradient test.
 * The function is implemented as PReLU layer in the file
 * parametric_relu.hpp
 *
 * @param input Input data used for evaluating the PReLU activation
 *   function.
 * @param target Target data used to evaluate the PReLU gradient.
 *
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
}*/

/**
 * Implementation of the Hard Shrink activation function test. The function is
 * implemented as Hard Shrink layer in the file hardshrink.hpp
 *
 * @param input Input data used for evaluating the Hard Shrink activation function.
 * @param target Target data used to evaluate the Hard Shrink activation.
 *
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
}*/

/**
 * Implementation of the HardShrink activation function derivative test.
 * The derivative function is implemented as HardShrink layer in the file
 * hardshrink.hpp
 *
 * @param input Input data used for evaluating the HardShrink activation
 * function.
 * @param target Target data used to evaluate the HardShrink activation.
 *
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
}*/

/**
 * Implementation of the Soft Shrink activation function test. The function is
 * implemented as Soft Shrink layer in the file softshrink.hpp.
 *
 * @param input Input data used for evaluating the Soft Shrink activation
 * function.
 * @param target Target data used to evaluate the Soft Shrink activation.
 *
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
}*/

/**
 * Implementation of the Soft Shrink activation function derivative test.
 * The derivative function is implemented as Soft Shrink layer in the file
 * softshrink.hpp
 *
 * @param input Input data used for evaluating the Soft Shrink activation
 * function.
 * @param target Target data used to evaluate the Soft Shrink activation.
 *
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
}*/

/**
 * Implementation of the ISRLU activation function test. The function is
 * implemented as ISRLU layer in the file isrlu.hpp.
 *
 * @param input Input data used for evaluating the ISRLU activation function.
 * @param target Target data used to evaluate the ISRLU activation.
 *
void CheckISRLUActivationCorrect(const arma::colvec input,
                                 const arma::colvec target)
{
  // Initialize ISRLU object with alpha = 1.0.
  ISRLU<> lrf(1.0);

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  lrf.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}*/

/**
 * Implementation of the ISRLU activation function derivative test. The function
 * is implemented as ISRLU layer in the file isrlu.hpp.
 *
 * @param input Input data used for evaluating the ISRLU activation function.
 * @param target Target data used to evaluate the ISRLU activation.
 *
void CheckISRLUDerivativeCorrect(const arma::colvec input,
                                 const arma::colvec target)
{
  // Initialize ISRLU object with alpha = 1.0.
  ISRLU<> lrf(1.0);

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
}*/

/**
 * Implementation of the Softmin activation function test. The function is
 * implemented as Softmin layer in the file softmin.hpp.
 *
 * @param input Input data used for evaluating the Softmin activation function.
 * @param target Target data used to evaluate the Softmin activation.
 *
void CheckSoftminActivationCorrect(const arma::colvec input,
                                   const arma::colvec target)
{
  // Initialize Softmin object.
  Softmin<> softmin;

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  softmin.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}*/

/**
 * Implementation of the Softmin activation function derivative test.
 * The function is implemented as Softmin layer in the file softmin.hpp.
 *
 * @param input Input data used for evaluating the Softmin activation function.
 * @param target Target data used to evaluate the Softmin activation.
 *
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
}*/

/**
 * Implementation of the Flatten T Swish activation function test. The function is
 * implemented as Flatten T Swish layer in the file flatten_t_swish.hpp.
 *
 * @param input Input data used for evaluating the Flatten T Swish activation
 *     function.
 * @param target Target data used to evaluate the Flatten T Swish activation.
 *
void CheckFlattenTSwishActivationCorrect(const arma::colvec input,
                                         const arma::colvec target)
{
  FlattenTSwish<> fts(0.4);
  arma::colvec activations;

  fts.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}*/

/**
 * Implementation of the Softmin activation function derivative test.
 * The function is implemented as Softmin layer in the file softmin.hpp.
 *
 * @param input Input data used for evaluating the Softmin activation function.
 * @param target Target data used to evaluate the Softmin activation.
 *
void CheckFlattenTSwishDerivateCorrect(const arma::colvec input,
                                       const arma::colvec target)
{
  FlattenTSwish<> fts;

  // Set the error to 1 to get the actual derivative.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);

  arma::colvec derivate;
  fts.Backward(input, error, derivate);
  for (size_t i = 0; i < derivate.n_elem; ++i)
  {
    REQUIRE(derivate.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
}*/

/**
 * Implementation of the ReLU6 activation function derivative test. The function
 * is implemented as ReLU6 layer in the file relu6.hpp.
 *
 * @param input Input data used for evaluating the ReLU6 activation function.
 * @param target Target data used to evaluate the ReLU6 activation.
 *
void CheckReLU6Correct(const arma::colvec input,
                       const arma::colvec ActivationTarget,
                       const arma::colvec DerivativeTarget)
{
  // Initialize ReLU6 object.
  ReLU6<> relu6;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives, activations;

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  relu6.Forward(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(ActivationTarget.at(i)).epsilon(1e-5));
  }
  relu6.Backward(activations, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(DerivativeTarget.at(i)).epsilon(1e-5));
  }
}*/

/**
 * Basic test of the ReLU6 function.
 *
TEST_CASE("ReLU6FunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("-2.0 3.0 0.0 6.0 24.0");

  // desiredActivations taken from PyTorch.
  const arma::colvec desiredActivations("0.0 3.0 0.0 6.0 6.0");

  // desiredDerivatives taken from PyTorch.
  const arma::colvec desiredDerivatives("0.0 1.0 0.0 0.0 0.0");

  CheckReLU6Correct(activationData, desiredActivations, desiredDerivatives);
}*/

/**
 * Basic test of the HardTanH function.
 *
TEST_CASE("HardTanHFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-1 1 1 -1 \
                                         1 -1 1 0");

  const arma::colvec desiredDerivatives("0 0 0 0 \
                                         1 1 0 1");

  CheckHardTanHActivationCorrect(activationData, desiredActivations);
  CheckHardTanHDerivativeCorrect(activationData, desiredDerivatives);
}*/

/**
 * Basic test of the PReLU function.
 *
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
}*/

/**
 * Basic test of the CReLU function.
 *
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
}*/

/**
 * Basic test of the swish function.
 *
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
}*/

/**
 * Basic test of the Hard Shrink function.
 *
TEST_CASE("HardShrinkFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-2 3.2 4.5 -100.2 1 -1 2 0");

  const arma::colvec desiredDerivatives("1 1 1 1 1 1 1 0");

  CheckHardShrinkActivationCorrect(activationData,
                                   desiredActivations);
  CheckHardShrinkDerivativeCorrect(desiredActivations,
                                   desiredDerivatives);
}*/

/**
 * Basic test of the Soft Shrink function.
 *
TEST_CASE("SoftShrinkFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-1.5 2.7 4 -99.7 0.5 -0.5 1.5 0");

  const arma::colvec desiredDerivatives("1 1 1 1 1 1 1 0");

  CheckSoftShrinkActivationCorrect(activationData,
                                   desiredActivations);
  CheckSoftShrinkDerivativeCorrect(desiredActivations,
                                   desiredDerivatives);
}*/

/**
 * Basic test of the CELU activation function.
 *
TEST_CASE("CELUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-0.86466472 3.2 4.5 \
                                         -1 1 -0.63212056 2 0");

  const arma::colvec desiredDerivatives("0.42119275 1 1 \
                                         0.36787944 1 \
                                         0.5314636 1 1");

  CheckCELUActivationCorrect(activationData, desiredActivations);
  CheckCELUDerivativeCorrect(desiredActivations, desiredDerivatives);
}*/

/**
 * Basic test of the ISRLU activation function.
 *
TEST_CASE("ISRLUFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec desiredActivations("-0.89442719 3.2 4.5 \
                                         -0.99995020 1 -0.70710678 2 0");

  const arma::colvec desiredDerivatives("0.41408666 1 1 \
                                         0.35357980 1 \
                                         0.54433105 1 1");

  CheckISRLUActivationCorrect(activationData, desiredActivations);
  CheckISRLUDerivativeCorrect(activationData, desiredDerivatives);
}*/

/**
 * Basic test of the Softmin function.
 *
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
}*/

/**
 * Basic test of Flatten T Swish function.
 *
TEST_CASE("FlattenTSwishFunctionTest", "[ActivationFunctionsTest]")
{
  // Random Value.
  arma::colvec input("-4.0 -1.0 2 3 4 5 6");

  // Hand Calculated and using PyTorch.
  arma::colvec desiredActivation(
      "0.4000000059604645 0.4000000059604645 2.1615941524505615 \
       3.2577223777770996 4.328054904937744 5.3665361404418945 \
       6.385164737701416");

  // Hand Calculated and using PyTorch.
  arma::colvec desiredDerivation("0.694792 0.694792 1.096893 1.079178 1.042602 \
                                  1.020182 1.009048");

  CheckFlattenTSwishActivationCorrect(input, desiredActivation);
  CheckFlattenTSwishDerivateCorrect(desiredActivation, desiredDerivation);
}*/
