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

#include <mlpack/methods/ann/ann.hpp>

#include "../catch.hpp"
#include "../test_catch_tools.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;

/**
 * Simple Huber Loss test.
 */
TEST_CASE("HuberLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output;
  arma::mat expectedOutput;
  double loss;
  HuberLoss module;

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
  REQUIRE(loss == Approx(6.36364).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
                Approx(-0.8032).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("-0.0401 -0.0833 -0.0501 -0.0833 0.0833 -0.0710 "
      "-0.0833 0.0646 0.0215 0.0833 0.0081 0.0833");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.530304.
  // Value calculated using torch.nn.SmoothL1Loss(reduction='mean').
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.530304).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
            Approx(-0.0669333).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Poisson Negative Log Likelihood Loss function test.
 */
TEST_CASE("PoissonNLLLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, input4, target4;
  arma::mat output1, output2, output3, output4;
  arma::mat expOutput1, expOutput2, expOutput3, expOutput4;
  PoissonNLLLoss module1(true, false, 1e-8, false);
  PoissonNLLLoss module2(true, true, 1e-08, true);
  PoissonNLLLoss module3(true, true, 1e-08, false);
  PoissonNLLLoss module4(false, true, 1e-08, false);

  // Test the Forward function on a user generated input.
  input = arma::mat("1.0 1.0 1.9 1.6 -1.9 3.7 -1.0 0.5");
  target = arma::mat("1.0 3.0 1.0 2.0 1.0 4.0 2.0 1.0");

  // Input required for module 4. Probs are in range [0, 1].
  input4 = arma::mat("0.658502 0.445627 0.667651 0.310549 \
                      0.589540 0.052568 0.549769 0.381504 ");
  target4 = arma::mat("1.0 3.0 1.0 2.0 1.0 4.0 2.0 1.0");

  double loss1 = module1.Forward(input, target);
  double loss2 = module2.Forward(input, target);
  double loss3 = module3.Forward(input, target);
  double loss4 = module4.Forward(input4, target4);
  REQUIRE(loss1 == Approx(4.8986).epsilon(1e-4));
  REQUIRE(loss2 == Approx(45.4139).epsilon(1e-4));
  REQUIRE(loss3 == Approx(5.6767).epsilon(1e-4));
  REQUIRE(loss4 == Approx(3.742157).epsilon(1e-4));

  // Test the Backward function.
  module1.Backward(input, target, output1);
  module2.Backward(input, target, output2);
  module3.Backward(input, target, output3);
  module4.Backward(input4, target4, output4);

  expOutput1 = arma::mat("0.214785 -0.0352148 0.710737 0.369129 \
                         -0.106304 4.55591 -0.204015 0.0810902");
  expOutput2 = arma::mat("1.71828 -0.281718 5.68589 2.95303\
                         -0.850431 36.4473 -1.63212 0.648721");
  expOutput3 = arma::mat("0.214785 -0.035215 0.710737 0.369129 \
                         -0.106304 4.555913 -0.204015 0.081090");
  expOutput4 = arma::mat("-0.064825 -0.716511 -0.062224 -0.680027 \
                          -0.087030 -9.386517 -0.329736 -0.202650");

  REQUIRE(output1.n_rows == input.n_rows);
  REQUIRE(output1.n_cols == input.n_cols);

  REQUIRE(output2.n_rows == input.n_rows);
  REQUIRE(output2.n_cols == input.n_cols);

  REQUIRE(output3.n_rows == input.n_rows);
  REQUIRE(output3.n_cols == input.n_cols);

  REQUIRE(output4.n_rows == input4.n_rows);
  REQUIRE(output4.n_cols == input4.n_cols);

  for (size_t i = 0; i < expOutput1.n_elem; ++i)
  {
    REQUIRE(output1[i] == Approx(expOutput1[i]).epsilon(1e-4));
    REQUIRE(output2[i] == Approx(expOutput2[i]).epsilon(1e-4));
    REQUIRE(output3[i] == Approx(expOutput3[i]).epsilon(1e-4));
    REQUIRE(output4[i] == Approx(expOutput4[i]).epsilon(1e-4));
  }
}

/**
 * Simple KL Divergence test.
 */
TEST_CASE("SimpleKLDivergenceTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output;
  arma::mat expectedOutput;
  double loss;
  KLDivergence module;

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
  REQUIRE(loss == Approx(2.33349).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == Approx(-5.8127).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("-0.0019 -0.0432 -0.0134 -0.0763 -0.0141 -0.0581 "
      "-0.0235 -0.0331 -0.0245 -0.0667 -0.0568 -0.0728");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 0.194458.
  // Value calculated using torch.nn.KLDivLoss(reduction='mean').
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.194458).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == Approx(-0.484392).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/*
 * Simple test for the mean squared logarithmic error function.
 */
TEST_CASE("SimpleMeanSquaredLogarithmicErrorTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output, expectedOutput;
  double loss;
  MeanSquaredLogarithmicError module;

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
  REQUIRE(loss == Approx(13.2728).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(-10.5619).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test the error function on a single input.
  input = arma::mat("2");
  target = arma::mat("3");
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.082760974810151655).epsilon(1e-3));

  // Test the Backward function on a single input.
  module.Backward(input, target, output);
  REQUIRE(arma::accu(output) == Approx(-0.1917880483011872).epsilon(1e-3));
  REQUIRE(output.n_elem == 1);
}

/*
 * Simple test for the mean squared error performance function.
 */
TEST_CASE("SimpleMeanSquaredErrorTest", "[LossFunctionsTest]")
{
  arma::mat input, output, target;
  MeanSquaredError module(false);

  // Test the Forward function on a user generated input and compare it against
  // the manually calculated result.
  input = arma::mat("1.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0");
  target = arma::zeros(1, 8);
  double error = module.Forward(input, target);
  REQUIRE(error == 0.5);

  // Test the Backward function.
  module.Backward(input, target, output);
  // We subtract a zero vector, so according to the used backward formula:
  // output = 2 * (input - target) / target.n_cols,
  // output * nofColumns / 2 should be equal to input.
  CheckMatrices(input, output * output.n_cols / 2);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test the error function on a single input.
  input = arma::mat("2");
  target = arma::mat("3");
  error = module.Forward(input, target);
  REQUIRE(error == 1.0);

  // Test the Backward function on a single input.
  module.Backward(input, target, output);
  // Test whether the output is negative.
  REQUIRE(arma::accu(output) == -2);
  REQUIRE(output.n_elem == 1);

  // Test for sum reduction
  module.Reduction() = true;

  // Test the Forward function
  error = module.Forward(input, target);
  REQUIRE(error == Approx(1.0).epsilon(1e-5));

  // Test the Backward function on a single input.
  module.Backward(input, target, output);
  // Test whether the output is negative.
  REQUIRE(arma::accu(output) == -2);
  REQUIRE(output.n_elem == 1);
}

/*
 * Simple test for the binary-cross-entropy lossfunction.
 */
TEST_CASE("SimpleBinaryCrossEntropyLossTest", "[LossFunctionsTest]")
{
  arma::mat input1, input2, input3, output, target1, target2, target3;
  BCELoss module1(1e-6, true);
  BCELoss module2(1e-6, false);
  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module1.Forward(input1, target1);
  REQUIRE(error1 - 8 * std::log(2) == Approx(0.0).margin(2e-5));

  input2 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5");
  target2 = arma::zeros(1, 6);
  input2.reshape(2, 3);
  target2.reshape(2, 3);
  double error2 = module2.Forward(input2, target2);
  REQUIRE(error2 - std::log(2) == Approx(0.0).margin(2e-5));

  input2 = arma::mat("0 1 1 0 1 0 0 1");
  target2 = arma::mat("0 1 1 0 1 0 0 1");
  double error3 = module1.Forward(input2, target2);
  REQUIRE(error3 == Approx(0.0).margin(1e-5));
  double error4 = module2.Forward(input2, target2);
  REQUIRE(error4 == Approx(0.0).margin(1e-5));

  // Test the Backward function.
  module1.Backward(input1, target1, output);
  for (double el : output)
  {
    // For the 0.5 constant vector we should get 1 / (1 - 0.5) = 2 everywhere.
    REQUIRE(el - 2 == Approx(0.0).margin(5e-6));
  }
  REQUIRE(output.n_rows == input1.n_rows);
  REQUIRE(output.n_cols == input1.n_cols);

  module1.Backward(input2, target2, output);
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (input2.at(i) == 0)
      REQUIRE(el - 1 == Approx(0.0).margin(2e-6));
    else
      REQUIRE(el + 1 == Approx(0.0).margin(2e-6));
  }
  REQUIRE(output.n_rows == input2.n_rows);
  REQUIRE(output.n_cols == input2.n_cols);
}

/**
 * Simple test for the Sigmoid Cross Entropy performance function.
 */
TEST_CASE("SimpleSigmoidCrossEntropyErrorTest", "[LossFunctionsTest]")
{
  arma::mat input1, input2, input3, output, target1,
            target2, target3, expectedOutput;
  SigmoidCrossEntropyError module;

  // Test the Forward function on a user generator input and compare it against
  // the calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  double error1 = module.Forward(input1, target1);
  double expected = 0.97407699;
  // Value computed using tensorflow.
  REQUIRE(error1 / input1.n_elem - expected == Approx(0.0).margin(1e-7));

  input2 = arma::mat("1 2 3 4 5");
  target2 = arma::mat("0 0 1 0 1");
  double error2 = module.Forward(input2, target2);
  expected = 1.5027283;
  REQUIRE(error2 / input2.n_elem - expected == Approx(0.0).margin(1e-6));

  input3 = arma::mat("0 -1 -1 0 -1 0 0 -1");
  target3 = arma::mat("0 -1 -1 0 -1 0 0 -1");
  double error3 = module.Forward(input3, target3);
  expected = 0.00320443;
  REQUIRE(error3 / input3.n_elem - expected == Approx(0.0).margin(1e-6));

  // Test the Backward function.
  module.Backward(input1, target1, output);
  expected = 0.62245929;
  for (size_t i = 0; i < output.n_elem; ++i)
    REQUIRE(output(i) - expected == Approx(0.0).margin(1e-5));
  REQUIRE(output.n_rows == input1.n_rows);
  REQUIRE(output.n_cols == input1.n_cols);

  expectedOutput = arma::mat(
      "0.7310586 0.88079709 -0.04742587 0.98201376 -0.00669285");
  module.Backward(input2, target2, output);
  for (size_t i = 0; i < output.n_elem; ++i)
    REQUIRE(output(i) - expectedOutput(i) == Approx(0.0).margin(1e-5));
  REQUIRE(output.n_rows == input2.n_rows);
  REQUIRE(output.n_cols == input2.n_cols);

  module.Backward(input3, target3, output);
  expectedOutput = arma::mat("0.5 1.2689414");
  for (size_t i = 0; i < 8; ++i)
  {
    double el = output.at(0, i);
    if (std::abs(input3.at(i) - 0.0) < 1e-5)
      REQUIRE(el - expectedOutput[0] == Approx(0.0).margin(2e-6));
    else
      REQUIRE(el - expectedOutput[1] == Approx(0.0).margin(2e-6));
  }
  REQUIRE(output.n_rows == input3.n_rows);
  REQUIRE(output.n_cols == input3.n_cols);
}

/**
 * Simple test for the Earth Mover Distance Layer.
 */
TEST_CASE("SimpleEarthMoverDistanceLayerTest", "[LossFunctionsTest]")
{
  arma::mat input1, input2, output, target1, target2, expectedOutput;
  arma::mat input3, target3;
  double loss;
  EarthMoverDistance module;

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input1 = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target1 = arma::zeros(1, 8);
  loss = module.Forward(input1, target1);
  double expected = 0.0;
  REQUIRE(loss / input1.n_elem - expected == Approx(0.0).margin(1e-7));

  input2 = arma::mat("1 2 3 4 5");
  target2 = arma::mat("1 0 1 0 1");
  loss = module.Forward(input2, target2);
  expected = -1.8;
  REQUIRE(loss / input2.n_elem - expected == Approx(0.0).margin(1e-6));

  // Test the Backward function.
  module.Backward(input1, target1, output);
  expected = 0.0;
  for (size_t i = 0; i < output.n_elem; ++i)
    REQUIRE(output(i) - expected == Approx(0.0).margin(1e-5));
  REQUIRE(output.n_rows == input1.n_rows);
  REQUIRE(output.n_cols == input1.n_cols);

  expectedOutput = arma::mat("-1 0 -1 0 -1");
  module.Backward(input2, target2, output);
  for (size_t i = 0; i < output.n_elem; ++i)
    REQUIRE(output(i) - expectedOutput(i) == Approx(0.0).margin(1e-5));
  REQUIRE(output.n_rows == input2.n_rows);
  REQUIRE(output.n_cols == input2.n_cols);

   // Test for mean reduction.
   module.Reduction() = false;
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
  loss = module.Forward(input3, target3);
  REQUIRE(loss == Approx(-0.00060089).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input3, target3, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == 
      Approx(-0.168867).epsilon(1e-3));
  REQUIRE(output.n_rows == input3.n_rows);
  REQUIRE(output.n_cols == input3.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/*
 * Mean Squared Error numerical gradient test.
 */
TEST_CASE("GradientMeanSquaredErrorTest", "[LossFunctionsTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::randu(2, 1);

      model = new FFN<MeanSquaredError, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(2);
      model->Add<Sigmoid>();
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

    FFN<MeanSquaredError, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/*
 * Reconstruction Loss numerical gradient test.
 */
TEST_CASE("GradientReconstructionLossTest", "[LossFunctionsTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input = arma::randu(10, 1);
      target = arma::randu(2, 1);

      model = new FFN<ReconstructionLoss, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(2);
      model->Add<Sigmoid>();
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

    FFN<ReconstructionLoss, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/*
 * Simple test for the dice loss function.
 */
TEST_CASE("DiceLossTest", "[LossFunctionsTest]")
{
  arma::mat input1, input2, target, output;
  double loss;
  DiceLoss module;

  // Test the Forward function. Loss should be 0 if input = target.
  input1 = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module.Forward(input1, target);
  REQUIRE(loss == Approx(0.0).margin(1e-5));

  // Test the Forward function. Loss should be 0.185185185.
  input2 = arma::ones(10, 1) * 0.5;
  loss = module.Forward(input2, target);
  REQUIRE(loss == Approx(0.185185185).epsilon(1e-5));

  // Test the Backward function for input = target.
  module.Backward(input1, target, output);
  for (double el : output)
  {
    // For input = target we should get 0.0 everywhere.
    REQUIRE(el == Approx(0.0).epsilon(1e-5));
  }
  REQUIRE(output.n_rows == input1.n_rows);
  REQUIRE(output.n_cols == input1.n_cols);

  // Test the Backward function.
  module.Backward(input2, target, output);
  for (double el : output)
  {
    // For the 0.5 constant vector we should get -0.0877914951989026 everywhere.
    REQUIRE(el == Approx(-0.0877914951989026).epsilon(1e-5));
  }
  REQUIRE(output.n_rows == input2.n_rows);
  REQUIRE(output.n_cols == input2.n_cols);
}

/*
 * Simple test for the mean bias error performance function.
 */
TEST_CASE("SimpleMeanBiasErrorTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output;
  double loss;
  MeanBiasError module;

  // Test for sum reduction.
  input = arma::mat("-0.0494 -1.1958 -1.0486 -0.2121 1.6028 0.0737 -0.7091 "
      "0.8612 0.9639 0.9648 0.0745 0.5924");
  target = arma::mat("0.4316 0.0164 -0.4478 1.1452 0.5106 0.9255 0.5571 0.0864 "
      "0.7059 -0.8288 -0.0231 -1.0526");

  input.reshape(4, 3);
  target.reshape(4, 3);

  // Test the forward function.
  // Loss should  be 0.1081.
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.1081).epsilon(1e-5));

  // Test the backward function.
  module.Backward(input, target, output);

  for(double el : output)
    REQUIRE(el == -1);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test for mean reduction by modifying
  // reduction parameter using accessor.
  module.Reduction() = false;

  // Test the forward function
  // loss should be 0.00900833
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.00900833).epsilon(1e-5));

  // Test the backward function
  module.Backward(input, target, output);

  for(double el : output)
    REQUIRE(el == Approx(-0.0833).epsilon(1e-3));
  REQUIRE(arma::accu(output) == Approx(-1).epsilon(1e-5));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
}

/**
 * Simple test for the Log-Hyperbolic-Cosine loss function.
 */
TEST_CASE("LogCoshLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output;
  double loss;
  LogCoshLoss module(2);

  // Test the Forward function. Loss should be 0 if input = target.
  input = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module.Forward(input, target);
  REQUIRE(loss == 0);

  // Test the Backward function for input = target.
  module.Backward(input, target, output);
  for (double el : output)
  {
    // For input = target we should get 0.0 everywhere.
    REQUIRE(el == Approx(0.0).epsilon(1e-5));
  }

  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test for sum reduction
  input = arma::mat("1 2 3 4 5");
  target = arma::mat("1 2.4 3.4 4.2 5.5");
  // Test the Forward function. Loss should be 0.546621.
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.546621).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::accu(output) == Approx(2.46962).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

    // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;

  // Test the Forward function. Loss should be 0.109324.
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.109324).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::accu(output) == Approx(0.49392).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
}

/**
 * Simple test for the Hinge Embedding loss function.
 */
TEST_CASE("HingeEmbeddingLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output, expectedOutput;
  double loss;
  HingeEmbeddingLoss module;

  // Test for sum reduction
  input = arma::mat("0.1778 0.0957 0.1397 0.2256 0.1203 0.2403 0.1925 0.3144 "
      "-0.2264 -0.3400 -0.3336 -0.8695");
  target = arma::mat("1 1 -1 1 1 -1 1 1 -1 1 1 1");
  expectedOutput = arma::mat("1 1 -1 1 1 -1 1 1 -1 1 1 1");
  input.reshape(4, 3);
  target.reshape(4, 3);
  expectedOutput.reshape(4, 3);

  // Test the forward function
  // Loss should be 2.4296
  // Value calculated using torch.nn.HingeEmbeddingLoss(reduction='sum')
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(2.4296).epsilon(1e-3));

  // Test the Backward function
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == Approx(6).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 1e-3);

  // Test for mean reduction by modifying reduction
  // parameter through the accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("0.0833 0.0833 -0.0833 0.0833 0.0833 -0.0833 "
    "0.0833 0.0833 -0.0833 0.0833 0.0833 0.0833");
  expectedOutput.reshape(4, 3);

  // Test the forward function
  // Loss should be 0.202467
  // Value calculated using torch.nn.HingeEmbeddingLoss(reduction='mean')
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(0.202467).epsilon(1e-3));

  // Test the backward function
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == Approx(0.5).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the l1 loss function.
 */
TEST_CASE("SimpleL1LossTest", "[LossFunctionsTest]")
{
  arma::mat input, output, target;
  double loss;
  L1Loss module(true);

  // Test the Forward function on a user generator input and compare it against
  // the manually calculated result.
  input = arma::mat("0.5 0.5 0.5 0.5 0.5 0.5 0.5");
  target = arma::zeros(1, 7);
  loss = module.Forward(input, target);
  // Value calculated using torch.nn.L1Loss(reduction='sum').
  REQUIRE(loss == 3.5);

  // Test the Backward function.
  module.Backward(input, target, output);
  for (double el : output)
    REQUIRE(el  == 1);

  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
}

/**
 * Simple test for the Cosine Embedding loss function.
 */
TEST_CASE("CosineEmbeddingLossTest", "[LossFunctionsTest]")
{
  arma::mat input1, input2, y, output;
  double loss;
  CosineEmbeddingLoss module;

  // Test the Forward function. Loss should be 0 if input1 = input2 and y = 1.
  input1 = arma::mat(1, 10);
  input2 = arma::mat(1, 10);
  input1.ones();
  input2.ones();
  y = arma::mat(1, 1);
  y.ones();
  loss = module.Forward(input1, input1);
  REQUIRE(loss == Approx(0.0).margin(1e-6));

  // Test the Backward function.
  module.Backward(input1, input1, output);
  REQUIRE(arma::accu(output) == Approx(0.0).margin(1e-6));

  // Check for dissimilarity.
  module.Similarity() = false;
  loss = module.Forward(input1, input1);
  REQUIRE(loss == Approx(1.0).epsilon(1e-4));

  // Test the Backward function.
  module.Backward(input1, input1, output);
  REQUIRE(arma::accu(output) == Approx(0.0).margin(1e-6));

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
  REQUIRE(loss == Approx(2.897367).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input1, input2, output);
  REQUIRE(arma::accu(output) == Approx(0.06324556).epsilon(1e-3));
}

/*
 * Simple test for the Margin Ranking Loss function.
 */
TEST_CASE("MarginRankingLossTest", "[LossFunctionsTest]")
{
  arma::mat input, input1, input2, target, output, expectedOutput;
  double loss;
  //  Test sum reduction
  MarginRankingLoss module;
  input1 = arma::mat("0.4287 -1.6208 -1.5006 -0.4473 1.5208 -4.5184 9.3574 "
      "-4.8090 4.3455 5.2070");
  input2 = arma::mat("-4.5288 -9.2766 -0.5882 -5.6643 -6.0175 8.8506 3.4759 "
      "-9.4886 2.2755 8.4951");
  expectedOutput = { { 0.0, 0.0,  1.0, 0.0,  1.0, -1.0, 0.0, 0.0,  1.0, -1.0 }, 
                     { 0.0, 0.0, -1.0, 0.0, -1.0,  1.0, 0.0, 0.0, -1.0,  1.0 } };
  input = arma::join_cols(input1, input2);
  target = arma::mat("1 1 -1 1 -1 1 1 1 -1 1");

  // Test the forward function
  // loss should be 30.3530 
  // value calculated using torch.nn.MarginRankingLoss(margin=1.0,reduction='sum')
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(30.3530).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;

  // Test the forward function
  // loss should be 3.0353
  // value calculated using torch.nn.MarginRankingLoss(margin=1.0,reduction='mean')
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(3.0353).epsilon(1e-5));

  // Test the backward function
  module.Backward(input, target, output);
  expectedOutput = { { 0.0, 0.0,  0.1, 0.0,  0.1, -0.1, 0.0, 0.0,  0.1, -0.1 },
                     { 0.0, 0.0, -0.1, 0.0, -0.1,  0.1, 0.0, 0.0, -0.1,  0.1 } };
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the Softmargin Loss function.
 */
TEST_CASE("SoftMarginLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output, expectedOutput;
  double loss;
  SoftMarginLoss module1;
  SoftMarginLoss module2(false);

  input = arma::mat("0.1778 0.0957 0.1397 0.1203 0.2403 0.1925 -0.2264 -0.3400 "
      "-0.3336");
  target = arma::mat("1 1 -1 1 -1 1 -1 1 1");
  input.reshape(3, 3);
  target.reshape(3, 3);

  // Test for sum reduction.

  // Calculated using torch.nn.SoftMarginLoss(reduction='sum').
  expectedOutput = arma::mat("-0.4557 -0.4761 0.5349 -0.4700 0.5598 -0.4520 "
      "0.4436 -0.5842 -0.5826");
  expectedOutput.reshape(3, 3);

  // Test the Forward function. Loss should be 6.41456.
  // Value calculated using torch.nn.SoftMarginLoss(reduction='sum').
  loss = module1.Forward(input, target);
  REQUIRE(loss == Approx(6.41456).epsilon(1e-3));

  // Test the Backward function.
  module1.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(-1.48227).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction.

  // Calculated using torch.nn.SoftMarginLoss(reduction='mean').
  expectedOutput = arma::mat("-0.0506 -0.0529 0.0594 -0.0522 0.0622 -0.0502 "
      "0.0493 -0.0649 -0.0647");
  expectedOutput.reshape(3, 3);

  // Test the Forward function. Loss should be 0.712729.
  // Value calculated using torch.nn.SoftMarginLoss(reduction='mean').
  loss = module2.Forward(input, target);
  REQUIRE(loss == Approx(0.712729).epsilon(1e-3));

  // Test the Backward function.
  module2.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(-0.164697).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}
/**
 * Simple test for the Mean Absolute Percentage Error function.
 */
TEST_CASE("MeanAbsolutePercentageErrorTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output, expectedOutput;
  MeanAbsolutePercentageError module;

  input = arma::mat("3 -0.5 2 7");
  target = arma::mat("2.5 0.2 2 8");
  expectedOutput = arma::mat("10.0 -125.0 12.5 -3.125");

  // Test the Forward function. Loss should be 95.625.
  // Loss value calculated manually.
  double loss = module.Forward(input, target);
  REQUIRE(loss == Approx(95.625).epsilon(1e-1));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(-105.625).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Test that the function that can access the parameters of the
 * VR Class Reward layer works.
 */
TEST_CASE("VRClassRewardLayerParametersTest", "[LossFunctionsTest]")
{
  // Parameter order : scale, sizeAverage.
  VRClassReward layer(2, false);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.Scale() == 2);
  REQUIRE(layer.SizeAverage() == false);
}

/*
 * Simple test for the Triplet Margin Loss function.
 */
TEST_CASE("TripletMarginLossTest")
{
  arma::mat anchor, positive, negative;
  arma::mat input, target, output;
  TripletMarginLoss module;

  // Test the Forward function on a user generated input and compare it against
  // the manually calculated result.
  anchor = arma::mat("2 3 5");
  positive = arma::mat("10 12 13");
  negative = arma::mat("4 5 7");

  input = { {2, 3, 5}, {10, 12, 13} };

  double loss = module.Forward(input, negative);
  REQUIRE(loss == 66);

  // Test the Backward function.
  module.Backward(input, negative, output);
  // According to the used backward formula:
  // output = 2 * (negative - positive) / anchor.n_cols,
  // output * nofColumns / 2 + positive should be equal to negative.
  CheckMatrices(negative, output * output.n_cols / 2 + positive);
  REQUIRE(output.n_rows == anchor.n_rows);
  REQUIRE(output.n_cols == anchor.n_cols);

  // Test the loss function on a single input.
  anchor = arma::mat("4");
  positive = arma::mat("7");
  negative = arma::mat("1");

  input = arma::mat(2, 1);
  input[0] = 4;
  input[1] = 7;

  loss = module.Forward(input, negative);
  REQUIRE(loss == 1.0);

  // Test the Backward function on a single input.
  module.Backward(input, negative, output);
  // Test whether the output is negative.
  REQUIRE(arma::accu(output) == -12);
  REQUIRE(output.n_elem == 1);
}

/**
 * Simple test for the Hinge loss function.
 */
TEST_CASE("HingeLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, target_b, output;
  double loss, loss_b;
  HingeLoss module1;
  HingeLoss module2(false);

  // Test the Forward function. Loss should be 0 if input = target.
  input = arma::ones(10, 1);
  target = arma::ones(10, 1);
  loss = module1.Forward(input, target);
  REQUIRE(loss == 0);

  // Test the Backward function for input = target.
  module1.Backward(input, target, output);
  for (double el : output)
  {
    // For input = target we should get 0.0 everywhere.
    REQUIRE(el == Approx(0.0).epsilon(1e-5));
  }

  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Randomly generated input.
  input = { { 0.90599973, -0.33040298, 0.07123354},
            { 0.71988434, 0.49657596, 0.39873373},
            { -0.57646927, 0.3951491 , -0.1003365},
            { 0.12528634, 0.68122971, 0.85448826} };

  // Randomly generated target.
  target = { { -1, -1, 1},
             { -1, 1, 1},
             { 1, -1, -1},
             { 1, -1, -1} };

  // Binary target can be obtained by replacing -1 with 0 in target.
  target_b = { { 0, 0, 1},
               { 0, 1, 1},
               { 1, 0, 0},
               { 1, 0, 0} };

  // Test for binary labels as target.
  loss = module1.Forward(input, target);
  loss_b = module1.Forward(input, target_b);

  // Loss should be same due to internal conversion of binary labels.
  REQUIRE(loss == loss_b);

  // Test for sum reduction.
  // Test the Forward function.
  // Loss calculated by referring to implementation of tf.keras.losses.hinge.
  loss = module1.Forward(input, target);
  REQUIRE(loss == Approx(14.61065).epsilon(1e-3));

  // Test the Backward function
  module1.Backward(input, target, output);
  REQUIRE(arma::accu(output) == Approx(-5).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test for mean reduction.
  // Test for the Forward function.
  // Loss calculated by referring to implementation of tf.keras.losses.hinge.
  loss = module2.Forward(input, target);
  REQUIRE(loss == Approx(1.21755).epsilon(1e-3));

  // Test the Backward function.
  module2.Backward(input, target, output);
  REQUIRE(arma::accu(output) == Approx(-0.41667).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
}

/**
 * Simple test for the MultiLabel Softmargin Loss function.
 */
TEST_CASE("MultiLabelSoftMarginLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output, expectedOutput;
  double loss;
  MultiLabelSoftMarginLoss module1;
  MultiLabelSoftMarginLoss module2(false);

  input = arma::mat("0.1778 0.0957 0.1397 0.1203 0.2403 0.1925 -0.2264 -0.3400 "
      "-0.3336");
  target = arma::mat("0 1 0 1 0 0 0 0 1");
  input.reshape(3, 3);
  target.reshape(3, 3);

  // Test for sum reduction.

  // Calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='sum').
  expectedOutput = arma::mat("0.1814 -0.1587 0.1783 -0.1567 0.1866 0.1827 "
      "0.1479 0.1386 -0.1942");
  expectedOutput.reshape(3, 3);

  // Test the Forward function. Loss should be 2.14829.
  // Value calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='sum').
  loss = module1.Forward(input, target);
  REQUIRE(loss == Approx(2.14829).epsilon(1e-5));

  // Test the Backward function.
  module1.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(0.505909).epsilon(1e-5));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction.

  // Calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='mean').
  expectedOutput = arma::mat("0.0605 -0.0529 0.0594 -0.0522 0.0622 0.0609 "
      "0.0493 0.0462 -0.0647");
  expectedOutput.reshape(3, 3);

  // Test the Forward function. Loss should be 0.716095.
  // Value calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='mean').
  loss = module2.Forward(input, target);
  REQUIRE(loss == Approx(0.716095).epsilon(1e-5));

  // Test the Backward function.
  module2.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(0.168636).epsilon(1e-5));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple test for the MultiLabel Softmargin Loss function.
 */
TEST_CASE("MultiLabelSoftMarginLossWeightedTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output, expectedOutput;
  arma::rowvec weights;
  double loss;
  weights = arma::mat("1 2 3");
  MultiLabelSoftMarginLoss module1(true, weights);
  MultiLabelSoftMarginLoss module2(false, weights);

  input = arma::mat("0.1778 0.0957 0.1397 0.2256 0.1203 0.2403 0.1925 0.3144 "
      "-0.2264 -0.3400 -0.3336 -0.8695");
  target = arma::mat("0 1 0 1 1 0 0 0 0 0 1 0");
  input.reshape(4, 3);
  target.reshape(4, 3);

  // Test for sum reduction.

  // Calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='sum').
  expectedOutput = arma::mat("0.1814 -0.1587 0.1783 -0.1479 -0.3133 0.3732 "
      "0.3653 0.3853 0.4436 0.4158 -0.5826 0.2954");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 5.35057.
  // Value calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='sum').
  loss = module1.Forward(input, target);
  REQUIRE(loss == Approx(5.35057).epsilon(1e-5));

  // Test the Backward function.
  module1.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(1.43577).epsilon(1e-5));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction.

  // Calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='mean').
  expectedOutput = arma::mat("0.0454 -0.0397 0.0446 -0.0370 -0.0783 0.0933 "
      "0.0913 0.0963 0.1109 0.1040 -0.1457 0.0738");
  expectedOutput.reshape(4, 3);

  // Test the Forward function. Loss should be 1.33764.
  // Value calculated using torch.nn.MultiLabelSoftMarginLoss(reduction='mean').
  loss = module2.Forward(input, target);
  REQUIRE(loss == Approx(1.33764).epsilon(1e-5));

  // Test the Backward function.
  module2.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) ==
      Approx(0.358943).epsilon(1e-5));
  REQUIRE(output.n_rows ==input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}

/**
 * Simple Negative Log Likelihood Loss test.
 */
TEST_CASE("NegativeLogLikelihoodLossTest", "[LossFunctionsTest]")
{
  arma::mat input, target, output;
  arma::mat expectedOutput;
  double loss;
  NegativeLogLikelihood module;

  // Test for sum reduction.
  input = arma::mat("-0.1689 -2.0033 -3.8886 -0.2862 -1.9392 -2.2532"
      " -1.0543 -0.6196 -2.1769 -1.2865 -1.4797 -0.7011");
  target = arma::mat("2 2 1 2");
  expectedOutput = arma::mat("0 0 -1.0000 0 0 -1.0000 0 -1.0000 0 0 0 -1.0000");
  input.reshape(3, 4);
  expectedOutput.reshape(3, 4);

  // Test the Forward function. Loss should be 7.4625.
  // Value calculated using torch.nn.NLLLoss(reduction='sum').
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(7.4625).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == Approx(-4).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);

  // Test for mean reduction by modifying reduction parameter using accessor.
  module.Reduction() = false;
  expectedOutput = arma::mat("0 0 -0.2500 0 0 -0.2500 0 -0.2500 0 0 0 -0.2500");
  expectedOutput.reshape(3, 4);

  // Test the Forward function. Loss should be 1.86562.
  // Value calculated using torch.nn.NLLLoss(reduction='mean').
  loss = module.Forward(input, target);
  REQUIRE(loss == Approx(1.86562).epsilon(1e-3));

  // Test the Backward function.
  module.Backward(input, target, output);
  REQUIRE(arma::as_scalar(arma::accu(output)) == Approx(-1).epsilon(1e-3));
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);
  CheckMatrices(output, expectedOutput, 0.1);
}
