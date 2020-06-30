/**
 * @file tests/loss_functions_test.cpp
 * @author Dakshit Agrawal
 * @author Sourabh Varshney
 * @author Atharva Khandait
 * @author Saksham Rastogi
 *
 * Tests for loss functions in mlpack::methods::ann:loss_functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/huber_loss.hpp>
#include <mlpack/methods/ann/loss_functions/kl_divergence.hpp>
#include <mlpack/methods/ann/loss_functions/earth_mover_distance.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/margin_ranking_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_logarithmic_error.hpp>
#include <mlpack/methods/ann/loss_functions/mean_bias_error.hpp>
#include <mlpack/methods/ann/loss_functions/dice_loss.hpp>
#include <mlpack/methods/ann/loss_functions/log_cosh_loss.hpp>
#include <mlpack/methods/ann/loss_functions/hinge_embedding_loss.hpp>
#include <mlpack/methods/ann/loss_functions/cosine_embedding_loss.hpp>
#include <mlpack/methods/ann/loss_functions/l1_loss.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(LossFunctionsTest);

/**
 * Simple Huber Loss test.
 */
BOOST_AUTO_TEST_CASE(HuberLossTest)
{
  arma::mat input, target, output;
  arma::mat expectedOutput;
  double loss;
  HuberLoss<> module;

  // Test for sum reduction.
  input = arma::mat("-0.0494 -1.1958 -1.0486 -0.2121 1.6028 0.0737 -0.7091 "
      "0.8612 0.9639 0.9648 0.0745 0.5924");
  target = arma::mat("0.4316 0.0164 -0.4478 1.1452 0.5106 0.9255 0.5571 0.0864 "
      "0.7059 -0.8288 -0.0231 -1.0526");
  expectedOutput = arma::mat("-0.4810 -1.0000 -0.6008 -1.0000 1.0000 -0.8518 "
      "-1.0000 0.7748 0.2580 1.0000 0.0976 1.0000");
  input.reshape(4, 3);
  target.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 6.36364.
  // Value calculated using torch.nn.SmoothL1Loss(reduction='sum').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 6.36364, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -0.8032, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("-0.0401 -0.0833 -0.0501 -0.0833 0.0833 -0.0710 "
      "-0.0833 0.0646 0.0215 0.0833 0.0081 0.0833");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.530304.
  // Value calculated using torch.nn.SmoothL1Loss(reduction='mean').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.530304, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -0.0669333, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/*
 * Simple test for the mean squared logarithmic error function.
 */
BOOST_AUTO_TEST_CASE(SimpleMeanSquaredLogarithmicErrorTest)
{
  arma::mat input, target, output, expectedOutput;
  double loss;
  MeanSquaredLogarithmicError<> module;

  // Test for sum reduction.
  input = arma::mat("-0.0494 1.1958 1.0486 -0.2121 1.6028 0.0737 -0.7091 "
      "0.8612 0.9639 0.9648 0.0745 0.5924");
  target = arma::mat("0.4316 0.0164 -0.4478 1.1452 0.5106 0.9255 0.5571 0.0864 "
      "0.7059 -0.8288 -0.0231 1.0526");
  expectedOutput = arma::mat("-0.8615 0.7016 1.2799 -2.5425 0.4181 -1.0880 "
      "-11.5339 0.5785 0.1434 2.4840 0.1772 -0.3188");
  input.reshape(4, 3);
  target.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 13.2728.
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 13.2728, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -10.5619, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("-0.0718 0.0585 0.1067 -0.2119 0.0348 -0.0907 "
      "-0.9612 0.0482 0.0120 0.2070 0.0148 -0.0266");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 1.10606.
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 1.10606, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -0.880156, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple KL Divergence test.
 */
BOOST_AUTO_TEST_CASE(KLDivergenceTest)
{
  arma::mat input, target, output;
  arma::mat expectedOutput;
  double loss;
  KLDivergence<> module;

  // Test for sum reduction.
  input = arma::mat("-0.7007 -2.0247 -0.7132 -0.4584 -0.2637 -1.1795 -0.1093 "
      "-1.0530 -2.4250 -0.4556 -0.7861 -0.9120");
  target = arma::mat("0.0223 0.5185 0.1610 0.9152 0.1689 0.6977 0.2823 0.3971 "
      "0.2939 0.8000 0.6816 0.8742");
  expectedOutput = arma::mat("-0.0223 -0.5185 -0.1610 -0.9152 -0.1689 -0.6977 "
      "-0.2823 -0.3971 -0.2939 -0.8000 -0.6816 -0.8742");
  input.reshape(4, 3);
  target.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 2.33349.
  // Value calculated using torch.nn.KLDivLoss(reduction='sum').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 2.33349, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -5.8127, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("-0.0019 -0.0432 -0.0134 -0.0763 -0.0141 -0.0581 "
      "-0.0235 -0.0331 -0.0245 -0.0667 -0.0568 -0.0728");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.194458.
  // Value calculated using torch.nn.KLDivLoss(reduction='mean').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.194458, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -0.484392, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}
/*
 * Simple test for the mean squared error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleMeanSquaredErrorTest)
{
  arma::mat input, output, target;
  MeanSquaredError<> module(false);

  // Test the Forward function on a user generated input and compare it against
  // the manually calculated result.
  input = arma::mat("1.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0");
  target = arma::zeros(1, 8);
  double error = module.Forward(input, target);
  BOOST_REQUIRE_EQUAL(error, 0.5);

  // Test the Backward function.
  module.Backward(input, target, output);
  // We subtract a zero vector, so according to the used backward formula:
  // output = 2 * (input - target) / target.n_cols,
  // output * nofColumns / 2 should be equal to input.
  CheckMatrices(input, output * output.n_cols / 2);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test the error function on a single input.
  input = arma::mat("2");
  target = arma::mat("3");
  error = module.Forward(input, target);
  BOOST_REQUIRE_EQUAL(error, 1.0);

  // Test the Backward function on a single input.
  module.Backward(input, target, output);
  // Test whether the output is negative.
  BOOST_REQUIRE_EQUAL(arma::accu(output), -2);
  BOOST_REQUIRE_EQUAL(output.n_elem, 1);

  // Test for sum reduction
  module.Reduction() = true;

  // Test the Forward function
  error = module.Forward(input, target);
  BOOST_REQUIRE_EQUAL(error, 1.0);

  // Test the Backward function on a single input.
  module.Backward(input, target, output);
  // Test whether the output is negative.
  BOOST_REQUIRE_EQUAL(arma::accu(output), -2);
  BOOST_REQUIRE_EQUAL(output.n_elem, 1);
}

/*
 * Simple test for the cross-entropy error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleCrossEntropyErrorTest)
{
  arma::mat input1, input2, output, target1, target2;
  arma::mat input3, input4, target3, target4, expectedOutput;
  double loss;
  CrossEntropyError<> module(1e-6);
  CrossEntropyError<> module1;
  CrossEntropyError<> module2(1e-10, false);

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(input1, target1);
  BOOST_REQUIRE_SMALL(error1 - 8 * std::log(2), 2e-5);

  input2 = arma::mat("0 1 1 0 1 0 0 1");
  target2 = arma::mat("0 1 1 0 1 0 0 1");
  double error2 = module.Forward(input2, target2);
  BOOST_REQUIRE_SMALL(error2, 1e-5);

  // Test the Backward function.
  module.Backward(input1, target1, output);
  for (double el : output)
  {
    // For the 0.5 constant vector we should get 1 / (1 - 0.5) = 2 everywhere.
    BOOST_REQUIRE_SMALL(el - 2, 5e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  module.Backward(input2, target2, output);
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (input2.at(i) == 0)
      BOOST_REQUIRE_SMALL(el - 1, 2e-6);
    else
      BOOST_REQUIRE_SMALL(el + 1, 2e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);

  // Example for Binary Classification with sum reduction.
  input3 = arma::mat("0.1778 0.0957 0.1397 0.2256 0.1203 0.2403 0.1925 0.3144");
  target3 = arma::mat("0 1 0 1 1 0 0 0");
  expectedOutput = arma::mat("1.2162 -10.4493 1.1624 -4.4326 -8.3126 1.3163 "
      "1.2384 1.4586");
  input3.reshape(4, 2);
  target3.reshape(4, 2);
  expectedOutput.reshape(4, 2);

  // Test the Forward function. Loss should be 7.16565.
  // Value calculated using torch.nn.BCELoss(reduction='sum').
  loss = module1.Forward(input3, target3);
  BOOST_REQUIRE_CLOSE(loss, 7.16565, 1e-3);

  // Test the Backward function.
  module1.Backward(input3, target3, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -16.8026, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input3.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input3.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test mean reduction by modifying reduction using accessor method.
  module1.Reduction() = false;
  expectedOutput = arma::mat("0.1520 -1.3062 0.1453 -0.5541 -1.0391 0.1645 "
      "0.1548 0.1823");
  expectedOutput.reshape(4, 2);

  // Test the Forward function. Loss should be 0.895706.
  // Value calculated using torch.nn.BCELoss(reduction='mean').
  loss = module1.Forward(input3, target3);
  BOOST_REQUIRE_CLOSE(loss, 0.895706, 1e-3);

  // Test the Backward function.
  module1.Backward(input3, target3, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -2.10032, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input3.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input3.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Example for Multi Class Classification with 3 classes , mean reduction.
  input4 = arma::mat("0.1778 0.0957 0.1397 0.2256 0.1203 0.2403 0.1925 0.3144 "
      "0.2264 0.3400 0.3336 0.8695");
  target4 = arma::mat("0 1 0 1 1 0 0 0 0 0 1 0");
  expectedOutput = arma::mat("0.1014 -0.8708 0.0969 -0.3694 -0.6927 0.1097 "
      "0.1032 0.1215 0.1077 0.1263 -0.2498 0.6386");
  input4.reshape(4, 3);
  target4.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.914338.
  // Value calculated using torch.nn.BCELoss(reduction='mean').
  loss = module2.Forward(input4, target4);
  BOOST_REQUIRE_CLOSE(loss, 0.914338, 1e-3);

  // Test the Backward function.
  module2.Backward(input4, target4, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -0.777462, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input4.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input4.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the Sigmoid Cross Entropy performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleSigmoidCrossEntropyErrorTest)
{
  arma::mat input1, input2, input3, output, target1,
            target2, target3, expectedOutput;
  SigmoidCrossEntropyError<> module;

  // Test the Forward function on a user generator input and compare it against
  // the calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(input1, target1);
  double expected = 0.97407699;
  // Value computed using tensorflow.
  BOOST_REQUIRE_SMALL(error1 / input1.n_elem - expected, 1e-7);

  input2 = arma::mat("1 2 3 4 5");
  target2 = arma::mat("0 0 1 0 1");
  double error2 = module.Forward(input2, target2);
  expected = 1.5027283;
  BOOST_REQUIRE_SMALL(error2 / input2.n_elem - expected, 1e-6);

  input3 = arma::mat("0 -1 -1 0 -1 0 0 -1");
  target3 = arma::mat("0 -1 -1 0 -1 0 0 -1");
  double error3 = module.Forward(input3, target3);
  expected = 0.00320443;
  BOOST_REQUIRE_SMALL(error3 / input3.n_elem - expected, 1e-6);

  // Test the Backward function.
  module.Backward(input1, target1, output);
  expected = 0.62245929;
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_SMALL(output(i) - expected, 1e-5);
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  expectedOutput = arma::mat(
      "0.7310586 0.88079709 -0.04742587 0.98201376 -0.00669285");
  module.Backward(input2, target2, output);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_SMALL(output(i) - expectedOutput(i), 1e-5);
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);

  module.Backward(input3, target3, output);
  expectedOutput = arma::mat("0.5 1.2689414");
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (std::abs(input3.at(i) - 0.0) < 1e-5)
      BOOST_REQUIRE_SMALL(el - expectedOutput[0], 2e-6);
    else
      BOOST_REQUIRE_SMALL(el - expectedOutput[1], 2e-6);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input3.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input3.n_cols);
}

/**
 * Simple test for the Earth Mover Distance Layer.
 */
BOOST_AUTO_TEST_CASE(SimpleEarthMoverDistanceLayerTest)
{
  arma::mat input1, input2, output, target1, target2, expectedOutput;
  arma::mat input3, target3;
  double loss;
  EarthMoverDistance<> module;
  EarthMoverDistance<> module2(false);

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(input1, target1);
  double expected = 0.0;
  BOOST_REQUIRE_SMALL(error1 / input1.n_elem - expected, 1e-7);

  input2 = arma::mat("1 2 3 4 5");
  target2 = arma::mat("1 0 1 0 1");
  double error2 = module.Forward(input2, target2);
  expected = -1.8;
  BOOST_REQUIRE_SMALL(error2 / input2.n_elem - expected, 1e-6);

  // Test the Backward function.
  module.Backward(input1, target1, output);
  expected = 0.0;
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_SMALL(output(i) - expected, 1e-5);
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  expectedOutput = arma::mat("-1 0 -1 0 -1");
  module.Backward(input2, target2, output);
  for (size_t i = 0; i < output.n_elem; i++)
    BOOST_REQUIRE_SMALL(output(i) - expectedOutput(i), 1e-5);
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);

  // Test for mean reduction.
  input3 = arma::mat("-0.0494 -1.1958 -1.0486 -0.2121 1.6028 0.0737 -0.7091 "
      "0.8612 0.9639 0.9648 0.0745 0.5924");
  target3 = arma::mat("0.4316 0.0164 -0.4478 1.1452 0.5106 0.9255 0.5571 "
      "0.0864 0.7059 -0.8288 -0.0231 -1.0526");
  expectedOutput = arma::mat("-0.0360 -0.0014 0.0373 -0.0954 -0.0426 -0.0771 "
      "-0.0464 -0.0072 -0.0588 0.0691 0.0019 0.0877");
  input3.reshape(4, 3);
  target3.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be -0.00060089.
  // Value calculated manually.
  loss = module2.Forward(input3, target3);
  BOOST_REQUIRE_CLOSE(loss, -0.00060089, 1e-3);

  // Test the Backward function.
  module2.Backward(input3, target3, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -0.168867, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input3.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input3.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/*
 * Mean Squared Error numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientMeanSquaredErrorTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::randu(2, 1);

      model = new FFN<MeanSquaredError<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 2);
      model->Add<SigmoidLayer<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      arma::mat output;
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<MeanSquaredError<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/*
 * Reconstruction Loss numerical gradient test.
 */
BOOST_AUTO_TEST_CASE(GradientReconstructionLossTest)
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::randu(2, 1);

      model = new FFN<ReconstructionLoss<>, NguyenWidrowInitialization>();
      model->Predictors() = input;
      model->Responses() = target;
      model->Add<IdentityLayer<> >();
      model->Add<Linear<> >(10, 2);
      model->Add<SigmoidLayer<> >();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      arma::mat output;
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<ReconstructionLoss<>, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  BOOST_REQUIRE_LE(CheckGradient(function), 1e-4);
}

/*
 * Simple test for the dice loss function.
 */
BOOST_AUTO_TEST_CASE(DiceLossTest)
{
  arma::mat input1, input2, target, output;
  double loss;
  DiceLoss<> module;

  // Test the Forward function. Loss should be 0 if input = target.
  input1 = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module.Forward(input1, target);
  BOOST_REQUIRE_SMALL(loss, 0.00001);

  // Test the Forward function. Loss should be 0.185185185.
  input2 = arma::ones(10, 1) * 0.5;
  loss = module.Forward(input2, target);
  BOOST_REQUIRE_CLOSE(loss, 0.185185185, 0.00001);

  // Test the Backward function for input = target.
  module.Backward(input1, target, output);
  for (double el : output)
  {
    // For input = target we should get 0.0 everywhere.
    BOOST_REQUIRE_CLOSE(el, 0.0, 0.00001);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input1.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input1.n_cols);

  // Test the Backward function.
  module.Backward(input2, target, output);
  for (double el : output)
  {
    // For the 0.5 constant vector we should get -0.0877914951989026 everywhere.
    BOOST_REQUIRE_CLOSE(el, -0.0877914951989026, 0.00001);
  }
  BOOST_REQUIRE_EQUAL(output.n_rows, input2.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input2.n_cols);
}

/*
 * Simple test for the mean bias error performance function.
 */
BOOST_AUTO_TEST_CASE(SimpleMeanBiasErrorTest)
{
  arma::mat input, target, output, expectedOutput;
  double loss;
  MeanBiasError<> module;

  // Test for sum reduction.
  input = arma::mat("-0.0494 -1.1958 -1.0486 -0.2121 1.6028 0.0737 -0.7091 "
      "0.8612 0.9639 0.9648 0.0745 0.5924");
  target = arma::mat("0.4316 0.0164 -0.4478 1.1452 0.5106 0.9255 0.5571 0.0864 "
      "0.7059 -0.8288 -0.0231 -1.0526");
  expectedOutput = arma::mat("-1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 "
      "-1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000");
  input.reshape(4, 3);
  target.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.1081.
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.1081, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -12, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("-0.0833 -0.0833 -0.0833 -0.0833 -0.0833 -0.0833 "
      "-0.0833 -0.0833 -0.0833 -0.0833 -0.0833 -0.0833");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.00900833.
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.00900833, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), -1, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the Log-Hyperbolic-Cosine loss function.
 */
BOOST_AUTO_TEST_CASE(LogCoshLossTest)
{
  arma::mat input, target, output;
  double loss;
  LogCoshLoss<> module(2);

  // Test the Forward function. Loss should be 0 if input = target.
  input = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module.Forward(input, target);
  BOOST_REQUIRE_EQUAL(loss, 0);

  // Test the Backward function for input = target.
  module.Backward(input, target, output);
  for (double el : output)
  {
    // For input = target we should get 0.0 everywhere.
    BOOST_REQUIRE_CLOSE(el, 0.0, 1e-5);
  }

  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test for sum reduction.
  input = arma::mat("1 2 3 4 5");
  target = arma::mat("1 2.4 3.4 4.2 5.5");

  // Test the Forward function. Loss should be 0.546621.
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.546621, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::accu(output), 2.46962, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;

  // Test the Forward function. Loss should be 0.109324.
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.109324, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::accu(output), 0.49392, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
}

/**
 * Simple test for the Hinge Embedding loss function.
 */
BOOST_AUTO_TEST_CASE(HingeEmbeddingLossTest)
{
  arma::mat input, target, output;
  arma::mat expectedOutput;
  double loss;
  HingeEmbeddingLoss<> module;

  // Test for sum reduction.
  input = arma::mat("0.1778 0.0957 0.1397 0.2256 0.1203 0.2403 0.1925 0.3144 "
      "-0.2264 -0.3400 -0.3336 -0.8695");
  target = arma::mat("1 1 -1 1 1 -1 1 1 -1 1 1 1");
  expectedOutput = arma::mat("1 1 -1 1 1 -1 1 1 -1 1 1 1");
  input.reshape(4, 3);
  target.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 2.4296.
  // Value calculated using torch.nn.HingeEmbeddingLoss(reduction='sum').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 2.4296, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), 6, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("0.0833 0.0833 -0.0833 0.0833 0.0833 -0.0833 "
      "0.0833 0.0833 -0.0833 0.0833 0.0833 0.0833");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.202467.
  // Value calculated using torch.nn.HingeEmbeddingLoss(reduction='mean').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(loss, 0.202467, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), 0.5, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the L1 loss function.
 */
BOOST_AUTO_TEST_CASE(SimpleL1LossTest)
{
  arma::mat input, output, target, expectedOutput;
  double loss;
  L1Loss<> module;

  // Test for sum reduction.
  input = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target = arma::zeros(1, 7);
  expectedOutput = arma::mat("1 1 1 1 1 1 1");

  // Test the Forward function. Loss should be 3.5.
  // Value calculated using torch.nn.L1Loss(reduction='sum').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_EQUAL(loss, 3.5);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), 7, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("0.1428 0.1428 0.1428 0.1428 0.1428 0.1428 "
      "0.1428");

  // Test the Forward function. Loss should be 0.5.
  // Value calculated using torch.nn.L1Loss(reduction='mean').
  loss = module.Forward(input, target);
  BOOST_REQUIRE_EQUAL(loss, 0.5);

  // Test the Backward function.
  module.Backward(input, target, output);
  BOOST_REQUIRE_CLOSE(arma::as_scalar(arma::accu(output)), 1, 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, input.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the Cosine Embedding loss function.
 */
BOOST_AUTO_TEST_CASE(CosineEmbeddingLossTest)
{
  arma::mat input1, input2, y, output;
  double loss;
  CosineEmbeddingLoss<> module;

  // Test the Forward function. Loss should be 0 if input1 = input2 and y = 1.
  input1 = arma::mat(1, 10);
  input2 = arma::mat(1, 10);
  input1.ones();
  input2.ones();
  y = arma::mat(1, 1);
  y.ones();
  loss = module.Forward(input1, input1);
  BOOST_REQUIRE_SMALL(loss, 1e-6);

  // Test the Backward function.
  module.Backward(input1, input1, output);
  BOOST_REQUIRE_SMALL(arma::accu(output), 1e-6);

  // Check for dissimilarity.
  module.Similarity() = false;
  loss = module.Forward(input1, input1);
  BOOST_REQUIRE_CLOSE(loss, 1.0, 1e-4);

  // Test the Backward function.
  module.Backward(input1, input1, output);
  BOOST_REQUIRE_SMALL(arma::accu(output), 1e-6);

  input1 = arma::mat(3, 2);
  input2 = arma::mat(3, 2);
  input1.fill(1);
  input1(4) = 2;
  input2.fill(1);
  input2(0) = 2;
  input2(1) = 2;
  input2(2) = 2;
  loss = module.Forward(input1, input2);
  // Calculated using torch.nn.CosineEmbeddingLoss().
  BOOST_REQUIRE_CLOSE(loss, 2.897367, 1e-3);

  // Test the Backward function.
  module.Backward(input1, input2, output);
  BOOST_REQUIRE_CLOSE(arma::accu(output), 0.06324556, 1e-3);

  // Check for correctness for cube.
  CosineEmbeddingLoss<> module2(0.5, true);

  arma::cube input3(3, 2, 2);
  arma::cube input4(3, 2, 2);
  input3.fill(1);
  input4.fill(1);
  input3(0) = 2;
  input3(1) = 2;
  input3(4) = 2;
  input3(6) = 2;
  input3(8) = 2;
  input3(10) = 2;
  input4(2) = 2;
  input4(9) = 2;
  input4(11) = 2;
  loss = module2.Forward(input3, input4);
  // Calculated using torch.nn.CosineEmbeddingLoss().
  BOOST_REQUIRE_CLOSE(loss, 0.55395, 1e-3);

  // Test the Backward function.
  module2.Backward(input3, input4, output);
  BOOST_REQUIRE_CLOSE(arma::accu(output), -0.36649111, 1e-3);

  // Check Output for mean type of reduction.
  CosineEmbeddingLoss<> module3(0.0, true, true);
  loss = module3.Forward(input3, input4);
  BOOST_REQUIRE_CLOSE(loss, 0.092325, 1e-3);

  // Check correctness for cube.
  module3.Similarity() = false;
  loss = module3.Forward(input3, input4);
  BOOST_REQUIRE_CLOSE(loss, 0.90767498236, 1e-3);

  // Test the Backward function.
  module3.Backward(input3, input4, output);
  BOOST_REQUIRE_CLOSE(arma::accu(output), 0.36649111, 1e-4);
}

/*
 * Simple test for the Margin Ranking Loss function.
 */
BOOST_AUTO_TEST_CASE(MarginRankingLossTest)
{
  arma::mat input, input1, input2, target, output;
  MarginRankingLoss<> module;

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("1 2 5 7 -1 -3");
  input2 = arma::mat("-1 3 -4 11 3 -3");
  input = arma::join_cols(input1, input2);
  target = arma::mat("1 -1 -1 1 -1 1");
  double error = module.Forward(input, target);
  // Computed using torch.nn.functional.margin_ranking_loss()
  BOOST_REQUIRE_CLOSE(error, 2.66667, 1e-3);

  // Test the Backward function.
  module.Backward(input, target, output);

  CheckMatrices(output, arma::mat("-0.000000 0.166667 -1.500000 0.666667 "
      "0.000000 -0.000000"), 1e-3);
  BOOST_REQUIRE_EQUAL(output.n_rows, target.n_rows);
  BOOST_REQUIRE_EQUAL(output.n_cols, target.n_cols);

  // Test the error function on another input.
  input1 = arma::mat("0.4287 -1.6208 -1.5006 -0.4473 1.5208 -4.5184 9.3574 "
      "-4.8090 4.3455 5.2070");
  input2 = arma::mat("-4.5288 -9.2766 -0.5882 -5.6643 -6.0175 8.8506 3.4759 "
      "-9.4886 2.2755 8.4951");
  input = arma::join_cols(input1, input2);
  target = arma::mat("1 1 -1 1 -1 1 1 1 -1 1");
  error = module.Forward(input, target);
  BOOST_REQUIRE_CLOSE(error, 3.03530, 1e-3);

  // Test the Backward function on the second input.
  module.Backward(input, target, output);

  CheckMatrices(output, arma::mat("0.000000 0.000000 0.091240 0.000000 "
      "-0.753830 1.336900 0.000000 0.000000 -0.207000 0.328810"), 1e-6);
}

BOOST_AUTO_TEST_SUITE_END();
