/**
 * @file tests/activation_functions_test.cpp
 * @author Marcus Edel
 * @author Dhawal Arora
 * @author Adam Kropp
 *
 * Tests for the various activation functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "./ann_test_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

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
        Approx(target.at(i)).epsilon(1e-5).margin(1e-15));
  }

  // Test the activation function using the entire vector as input.
  arma::colvec activations;
  ActivationFunction::Fn(input, activations);
  for (size_t i = 0; i < activations.n_elem; ++i)
  {
    REQUIRE(activations.at(i) ==
        Approx(target.at(i)).epsilon(1e-5).margin(1e-15));
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
    double output = ActivationFunction::Fn(input.at(i));
    REQUIRE(ActivationFunction::Deriv(input.at(i), output) ==
        Approx(target.at(i)).epsilon(1e-5).margin(1e-15));
  }

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec outputs;
  ActivationFunction::Fn(input, outputs);
  arma::colvec derivatives;
  ActivationFunction::Deriv(input, outputs, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5).margin(1e-15));
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
        Approx(input.at(i)).epsilon(1e-5).margin(1e-15));
  }

  // Test the calculation of the inverse using the entire vector as input.
  arma::colvec activations;
  ActivationFunction::Fn(input, activations);
  ActivationFunction::Inv(activations, activations);

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    REQUIRE(activations.at(i) == Approx(input.at(i)).epsilon(1e-5).margin(1e-15));
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
  LeakyReLU lrf;

  // Test the activation function using the entire vector as input.
  arma::colvec activations(input.n_elem);
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
  LeakyReLU lrf;

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives(input.n_elem);

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  arma::colvec output(input.n_elem);
  lrf.Forward(input, output);
  lrf.Backward(input, output, error, derivatives);
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
  // Initialize ELU object with alpha = 1.
  ELU module(1);
  module.Training() = true;
  module.InputDimensions() = std::vector<size_t>({ input.n_elem });
  module.ComputeOutputDimensions();

  // Test the activation function using the entire vector as input.
  arma::colvec activations(input.n_elem);
  module.Forward(input, activations);
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
  ELU module(1);
  module.Training() = true;
  module.InputDimensions() = std::vector<size_t>({ input.n_elem });
  module.ComputeOutputDimensions();

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives(input.n_elem), activations(input.n_elem);

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  module.Forward(input, activations);
  module.Backward(input, activations, error, derivatives);
  for (size_t i = 0; i < derivatives.n_elem; ++i)
  {
    REQUIRE(derivatives.at(i) == Approx(target.at(i)).epsilon(1e-5));
  }
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
  // Initialize CELU object with alpha = 1.
  CELU module(1);
  module.Training() = true;
  module.InputDimensions() = std::vector<size_t>({ input.n_elem });
  module.ComputeOutputDimensions();

  // Test the activation function using the entire vector as input.
  arma::colvec activations(input.n_elem);
  module.Forward(input, activations);
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
  CELU module(1);
  module.Training() = true;
  module.InputDimensions() = std::vector<size_t>({ input.n_elem });
  module.ComputeOutputDimensions();

  // Test the calculation of the derivatives using the entire vector as input.
  arma::colvec derivatives(input.n_elem), activations(input.n_elem);

  // This error vector will be set to 1 to get the derivatives.
  arma::colvec error = arma::ones<arma::colvec>(input.n_elem);
  module.Forward(input, activations);
  module.Backward(input, activations, error, derivatives);
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
  arma::mat output(arma::size(input));
  
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
  arma::mat output(arma::size(input));
  
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
 */
TEST_CASE("SELUFunctionDerivativeTest", "[ActivationFunctionsTest]")
{
  arma::mat input = arma::ones<arma::mat>(1000, 1);
  arma::mat error = arma::ones<arma::mat>(input.n_elem, 1);
  arma::mat derivatives(arma::size(input)), activations(arma::size(input));

  SELU selu;
  selu.Training() = true;
  selu.InputDimensions() = std::vector<size_t>({ input.n_elem });
  selu.ComputeOutputDimensions();

  selu.Forward(input, activations);
  selu.Backward(input, activations, error, derivatives);
  
  REQUIRE(arma::as_scalar(arma::abs(arma::mean(derivatives) -
      selu.Lambda())) <= 10e-4);

  input.fill(-1);

  selu.Forward(input, activations);
  selu.Backward(input, activations, error, derivatives);

  REQUIRE(arma::as_scalar(arma::abs(arma::mean(derivatives) -
      selu.Lambda() * selu.Alpha() - arma::mean(activations))) <= 10e-4);
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
  CheckDerivativeCorrect<TanhFunction>(activationData, desiredDerivatives);
  CheckInverseCorrect<TanhFunction>(desiredActivations);
  
  arma::mat inputTemp(desiredActivations.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<TanhFunction>(inputTemp,-2,2);
  REQUIRE(maxRelativeError <= 1e-3);
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
  CheckDerivativeCorrect<LogisticFunction>(activationData,
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
  CheckDerivativeCorrect<SoftsignFunction>(activationData,
                                           desiredDerivatives);
  CheckInverseCorrect<SoftsignFunction>(desiredActivations);

  arma::mat inputTemp(desiredActivations.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<SoftsignFunction>(inputTemp,-1,1);
  REQUIRE(maxRelativeError <= 1e-3);
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
  CheckDerivativeCorrect<RectifierFunction>(activationData,
                                            desiredDerivatives);

  arma::mat inputTemp(desiredActivations.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<RectifierFunction>(inputTemp,-4,4);
  REQUIRE(maxRelativeError <= 1e-3);  
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
 * Basic test of the softplus function.
 */
TEST_CASE("SoftplusFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("-2 3.2 4.5 -100.2 1 -1 2 0 1000 10000");

  const arma::colvec desiredActivations("0.12692801 3.23995333 4.51104774 \
                                         0 1.31326168 0.31326168 2.12692801 \
                                         0.69314718 1000 10000");

  const arma::colvec desiredDerivatives("0.11920292 0.96083427 0.98901305 \
                                        3.04574060e-44 0.73105857 0.26894142 \
                                        0.88079707 0.5 1 1");

  CheckActivationCorrect<SoftplusFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<SoftplusFunction>(activationData,
                                           desiredDerivatives);
  CheckInverseCorrect<SoftplusFunction>(desiredActivations);
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
  CheckDerivativeCorrect<HardSigmoidFunction>(activationData,
                                              desiredDerivatives);

  arma::mat inputTemp(desiredActivations.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<HardSigmoidFunction>(inputTemp,-4,4);
  REQUIRE(maxRelativeError <= 1e-3);  
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

  const arma::colvec desiredDerivatives("-0.10835509242 1.01574465162 \
                                        1.00190704417 -3.05183208657e-42 \
                                        1.0490362201 0.0592167558774 \
                                        1.06931793428 0.6");

  CheckActivationCorrect<MishFunction>(activationData,
                                       desiredActivations);
  CheckDerivativeCorrect<MishFunction>(activationData,
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

  const arma::colvec desiredDerivatives("-1.1053292 1.0178798 1.001974 \
                                        -1.0 1.1815685 -1.1815685 1.1053292 \
                                        0.0");

  CheckActivationCorrect<LiSHTFunction>(activationData,
                                        desiredActivations);
  CheckDerivativeCorrect<LiSHTFunction>(activationData,
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

  const arma::colvec desiredDerivatives("-0.086099065 1.0064931 1.0000293 \
                                        0 1.0829639 -0.0829639 1.0860991 0.5");

  CheckActivationCorrect<GELUFunction>(activationData,
                                       desiredActivations);
  CheckDerivativeCorrect<GELUFunction>(activationData,
                                       desiredDerivatives);
}

/**
 * Basic test of the Elliot function.
 */
TEST_CASE("ElliotFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("-1.0 0 1.2 2.1 2.8 1.0 -1.1 -2.0 -3.1");

  // Hand-calculated values.
  arma::colvec desiredActivation(
      "-0.5 0.0 0.5454545454545454 0.6774193548387097 0.7368421052631579 \
      0.5 -0.5238095238095238 -0.6666666666666666, -0.7560975609756099");

  // Hand-calculated values.
  arma::colvec desiredDerivate(
      "0.25 1.0 0.20661157024793392 0.10405827263267425 0.06925207756232689 \
      0.25 0.22675736961451246 0.11111111111111113 0.05948839976204635");

  CheckActivationCorrect<ElliotFunction>(activationData
                                                ,desiredActivation);
  CheckDerivativeCorrect<ElliotFunction>(activationData
                                                ,desiredDerivate);
  arma::mat inputTemp(activationData.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<ElliotFunction>(inputTemp);
  REQUIRE(maxRelativeError <= 1e-3);
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

  const arma::colvec desiredDerivatives("-0.074651888 1.0812559 1.0379111 \
                                        0.0 0.92767051 -0.025344425 1.0907842 \
                                        0.5");

  CheckActivationCorrect<ElishFunction>(activationData,
                                        desiredActivations);
  CheckDerivativeCorrect<ElishFunction>(activationData,
                                        desiredDerivatives);
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

  const arma::colvec desiredDerivatives("0.16 -0.05065792 -0.019930796 \
                                        1.9876519e-06 -0.5 0.5 -0.16 0.0");

  CheckActivationCorrect<InvQuadFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<InvQuadFunction>(activationData,
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

  const arma::colvec desiredDerivatives("-4.0 6.4 9.0 -200.4 \
                                        2.0 -2.0 4.0 0.0");

  CheckActivationCorrect<QuadraticFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<QuadraticFunction>(activationData,
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

  const arma::colvec desiredDerivatives("5.7277825 11.622636 19.024551 \
                                        1024.4765 1.8862944 1.8862944 \
                                        5.7277825 0.0");

  CheckActivationCorrect<SplineFunction>(activationData1, desiredActivations);
  CheckDerivativeCorrect<SplineFunction>(activationData1,
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

  const arma::colvec desiredDerivatives("-0.89442719 0.95447998 0.97618706 \
                                        -0.9999502 0.70710678 -0.70710678 \
                                        0.89442719 0.0");

  CheckActivationCorrect<MultiQuadFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<MultiQuadFunction>(activationData,
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

  const arma::colvec desiredDerivatives("29.556224 -0.048914645 -0.027772491 \
                                        -0.020213841 0.36787944 8.1548455 0.0 \
                                        2.0");

  CheckActivationCorrect<Poisson1Function>(activationData1, desiredActivations);
  CheckDerivativeCorrect<Poisson1Function>(activationData1,
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

  const arma::colvec desiredDerivatives("0.073262556 -0.00022856224 \
                                        -1.4447052e-08 0.0 -0.73575888 \
                                        0.73575888 -0.073262556 0.0");

  CheckActivationCorrect<GaussianFunction>(activationData,
                                           desiredActivations);
  CheckDerivativeCorrect<GaussianFunction>(activationData,
                                           desiredDerivatives);
}

/**
 * Basic test of the Hard Swish function.
 */
TEST_CASE("HardSwishFunctionTest", "[ActivationFunctionsTest]")
{
  // Randomly generated data.
  const arma::colvec activationData("3.6544 -1.9714 -5.2277 1.5448 2.1164");

  // Hand-calculated values.
  const arma::colvec desiredActivations("3.6544 -0.3379636 0.0 \
                                         1.1701345 1.8047248");

  // Hand-calculated values.
  const arma::colvec desiredDerivatives("1.0 -0.15713333 0.0 1.0149333 \
                                        1.2054667");

  CheckActivationCorrect<HardSwishFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<HardSwishFunction>
      (activationData, desiredDerivatives);
}

/**
 * Basic test of the TanhExp function.
 */
TEST_CASE("TanhExpFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("-2 3.2 4.5 1 -1 2 0");

  // Hand-calculated values.
  const arma::colvec desiredActivations("-0.26903 3.20000 4.50000 \
                                         0.991329 -0.352135 2.0 0.0000");

  // Hand-calculated values.
  const arma::colvec desiredDerivatives("-0.13125793 1.0 1.0 1.0382654 \
                                        0.029872881 1.0000218 0.76159416");

  CheckActivationCorrect<TanhExpFunction>(activationData, desiredActivations);
  CheckDerivativeCorrect<TanhExpFunction>(activationData,
      desiredDerivatives);
}

/**
 * Basic test of the SILU(Sigmoid Weighted Linear Unit) Function
 */
TEST_CASE("SILUFunctionTest", "[ActivationFunctionsTest]")
{
  // Random generated values.
  const arma::colvec activationData("-2 2 4.5 -5.7 -1 1 0 10");

  // Calculated with PyTorch.
  arma::colvec desiredActivation(
      "-0.23840583860874176 1.7615940570831299 4.450558662414551 \
       -0.01900840364396572 -0.2689414322376251 0.7310585975646973 \
       0.0 9.99954605102539");

  // Calculated with PyTorch.
  arma::colvec desiredDerivate(
      "-0.090784249 1.0907842 1.0379111 -0.015610205 0.072329488 \
      0.92767051 0.5 1.0004086");

  CheckActivationCorrect<SILUFunction>(activationData, desiredActivation);
  CheckDerivativeCorrect<SILUFunction>(activationData, desiredDerivate);
}

/**
 * Basic test of the SILU(Sigmoid Weighted Linear Unit) Function
 */
TEST_CASE("SwishFunctionTest", "[ActivationFunctionsTest]")
{
    // Random generated values.
    const arma::colvec activationData("-2 2 4.5 -5.7 -1 1 0 10");

    // Calculated with PyTorch.
    arma::colvec desiredActivation(
            "-0.23840583860874176 1.7615940570831299 4.450558662414551 \
       -0.01900840364396572 -0.2689414322376251 0.7310585975646973 \
       0.0 9.99954605102539");

    // Calculated with PyTorch.
    arma::colvec desiredDerivate(
            "-0.090784249 1.0907842 1.0379111 -0.015610205 0.072329488 \
      0.92767051 0.5 1.0004086");

    CheckActivationCorrect<SwishFunction>(activationData, desiredActivation);
    CheckDerivativeCorrect<SwishFunction>(activationData, desiredDerivate);
}

/**
 * Basic test of the Hyper-Sinh Function
 */
TEST_CASE("HyperSinhFunctionTest", "[ActivationFunctionsTest]")
{
  arma::colvec activationData("-1.0 0 1.2 1.5 1.8 1.0 -1.1 -1.6 -1.8");

  //Hand calculated values
	arma::colvec desiredActivation(
		"-0.25 0.0 0.5031537851373908 0.7097598183649391 0.9807247626985599 \
		0.3917337312146005 -0.3327500000000001 -1.0240000000000002 \
		-1.4580000000000002");

	// Hand-calculated values.
	arma::colvec desiredDerivate(
		"0.75 0.0 0.6035518557747915 0.7841365384144158 1.0358243921057555 \
		0.5143602116050813 0.9075000000000002 1.9200000000000004 2.43");
  
  CheckActivationCorrect<HyperSinhFunction>(activationData, desiredActivation);
  CheckDerivativeCorrect<HyperSinhFunction>(activationData, desiredDerivate);

  arma::mat inputTemp(activationData.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<HyperSinhFunction>(inputTemp);
  REQUIRE(maxRelativeError <= 1e-3);
}

/**
 * Basic test of the Bipolar Sigmoid Function
 */
TEST_CASE("BipolaSigmoidFunctionTest", "[ActivationFunctionsTest]")
{
  const arma::colvec activationData("-1.0 0 1.2 2.1 2.8 1.0 -1.1 -2.0 -3.1");

  // Hand-calculated values.
  arma::colvec desiredActivation(
      "-0.4621171572 0.0 0.5370495669 0.7818063576 0.8853516482 \
      0.462117157 -0.50052021119 -0.7615941559 -0.9137854901");

  // Hand-calculated values.
  arma::colvec desiredDerivate(
      "0.3932238665 0.5 0.3557888813 0.1943894096 0.1080762295 \
      0.3932238666 0.3747397590 0.2099871708 0.0824980390");

  CheckActivationCorrect<BipolarSigmoidFunction>(activationData
                                                ,desiredActivation);
  CheckDerivativeCorrect<BipolarSigmoidFunction>(activationData
                                                ,desiredDerivate);

  arma::mat inputTemp(activationData.n_elem, 1);
  double maxRelativeError = ActivationJacobianTest<BipolarSigmoidFunction>(inputTemp,-2,2);
  REQUIRE(maxRelativeError <= 1e-3);
}