
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_activation_functions_test.cpp:

Program Listing for File activation_functions_test.cpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_activation_functions_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/activation_functions_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
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
   #include <mlpack/methods/ann/activation_functions/hard_swish_function.hpp>
   #include <mlpack/methods/ann/activation_functions/tanh_exponential_function.hpp>
   #include <mlpack/methods/ann/activation_functions/silu_function.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::ann;
   
   // Generate dataset for activation function tests.
   const arma::colvec activationData("-2 3.2 4.5 -100.2 1 -1 2 0");
   
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
   }
   
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
   }
   
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
   }
   
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
   }
   
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
   }
   
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
   
   TEST_CASE("IdentityFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredDerivatives = arma::ones<arma::colvec>(
         activationData.n_elem);
   
     CheckActivationCorrect<IdentityFunction>(activationData, activationData);
     CheckDerivativeCorrect<IdentityFunction>(activationData, desiredDerivatives);
   }
   
   TEST_CASE("RectifierFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("0 3.2 4.5 0 1 0 2 0");
   
     const arma::colvec desiredDerivatives("0 1 1 0 1 0 1 0");
   
     CheckActivationCorrect<RectifierFunction>(activationData, desiredActivations);
     CheckDerivativeCorrect<RectifierFunction>(desiredActivations,
                                               desiredDerivatives);
   }
   
   TEST_CASE("LeakyReLUFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("-0.06 3.2 4.5 -3.006 \
                                            1 -0.03 2 0");
   
     const arma::colvec desiredDerivatives("0.03 1 1 0.03 \
                                            1 0.03 1 1");
   
     CheckLeakyReLUActivationCorrect(activationData, desiredActivations);
     CheckLeakyReLUDerivativeCorrect(desiredActivations, desiredDerivatives);
   }
   
   TEST_CASE("HardTanHFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("-1 1 1 -1 \
                                            1 -1 1 0");
   
     const arma::colvec desiredDerivatives("0 0 0 0 \
                                            1 1 0 1");
   
     CheckHardTanHActivationCorrect(activationData, desiredActivations);
     CheckHardTanHDerivativeCorrect(activationData, desiredDerivatives);
   }
   
   TEST_CASE("ELUFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("-0.86466471 3.2 4.5 -1.0 \
                                            1 -0.63212055 2 0");
   
     const arma::colvec desiredDerivatives("0.13533529 1 1 0 \
                                            1 0.36787945 1 1");
   
     CheckELUActivationCorrect(activationData, desiredActivations);
     CheckELUDerivativeCorrect(activationData, desiredDerivatives);
   }
   
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
   
   TEST_CASE("HardShrinkFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("-2 3.2 4.5 -100.2 1 -1 2 0");
   
     const arma::colvec desiredDerivatives("1 1 1 1 1 1 1 0");
   
     CheckHardShrinkActivationCorrect(activationData,
                                      desiredActivations);
     CheckHardShrinkDerivativeCorrect(desiredActivations,
                                      desiredDerivatives);
   }
   
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
   
   TEST_CASE("SoftShrinkFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("-1.5 2.7 4 -99.7 0.5 -0.5 1.5 0");
   
     const arma::colvec desiredDerivatives("1 1 1 1 1 1 1 0");
   
     CheckSoftShrinkActivationCorrect(activationData,
                                      desiredActivations);
     CheckSoftShrinkDerivativeCorrect(desiredActivations,
                                      desiredDerivatives);
   }
   
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
   
   TEST_CASE("ISRLUFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec desiredActivations("-0.89442719 3.2 4.5 \
                                            -0.99995020 1 -0.70710678 2 0");
   
     const arma::colvec desiredDerivatives("0.41408666 1 1 \
                                            0.35357980 1 \
                                            0.54433105 1 1");
   
     CheckISRLUActivationCorrect(activationData, desiredActivations);
     CheckISRLUDerivativeCorrect(activationData, desiredDerivatives);
   }
   
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
   
   TEST_CASE("HardSwishFunctionTest", "[ActivationFunctionsTest]")
   {
     // Randomly generated data.
     const arma::colvec activationData("3.6544 -1.9714 -5.2277 1.5448 2.1164");
   
     // Hand-calculated values.
     const arma::colvec desiredActivations("3.6544 -0.3379636 0.0 \
                                            1.1701345 1.8047248");
   
     // Hand-calculated values.
     const arma::colvec desiredDerivatives("1.0 0.38734546 0.5 \
                                            0.89004483 1.1015749");
   
     CheckActivationCorrect<HardSwishFunction>(activationData, desiredActivations);
     CheckDerivativeCorrect<HardSwishFunction>
         (desiredActivations, desiredDerivatives);
   }
   
   TEST_CASE("TanhExpFunctionTest", "[ActivationFunctionsTest]")
   {
     const arma::colvec activationData("-2 3.2 4.5 1 -1 2 0");
   
     // Hand-calculated values.
     const arma::colvec desiredActivations("-0.26903 3.20000 4.50000 \
                                            0.991329 -0.352135 2.0 0.0000");
   
     // Hand-calculated values.
     const arma::colvec desiredDerivatives("0.523051 1.0000 1.0000 \
                                            1.03924 0.449818 1.00002 0.761594");
   
     CheckActivationCorrect<TanhExpFunction>(activationData, desiredActivations);
     CheckDerivativeCorrect<TanhExpFunction>(desiredActivations,
         desiredDerivatives);
   }
   
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
         "0.38191673159599304 1.073788046836853 1.0392179489135742 \
          0.49049633741378784 0.36713290214538574 0.8354039788246155 \
          0.5 1.0004087686538696");
   
     CheckActivationCorrect<SILUFunction>(activationData, desiredActivation);
     CheckDerivativeCorrect<SILUFunction>(desiredActivation, desiredDerivate);
   }
   
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
   }
