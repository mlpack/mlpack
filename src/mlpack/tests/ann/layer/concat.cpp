/**
 * @file tests/ann/layer/concat.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 *
 * Tests the ann layer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * Simple concat module test.
 */
TEST_CASE("SimpleConcatLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta, error;

  Linear* moduleA = new Linear(10);
  moduleA->InputDimensions() = std::vector<size_t>({ 10 });
  moduleA->ComputeOutputDimensions();
  arma::mat weightsA(moduleA->WeightSize(), 1);
  moduleA->SetWeights((double*) weightsA.memptr());
  moduleA->Parameters().randu();

  Linear* moduleB = new Linear(10);
  moduleB->InputDimensions() = std::vector<size_t>({ 10 });
  moduleB->ComputeOutputDimensions();
  arma::mat weightsB(moduleB->WeightSize(), 1);
  moduleB->SetWeights((double*) weightsB.memptr());
  moduleB->Parameters().randu();

  Concat module;
  module.Add(moduleA);
  module.Add(moduleB);
  module.InputDimensions() = std::vector<size_t>({ 10 });
  module.ComputeOutputDimensions();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  output.set_size(module.OutputSize(), 1);
  module.Forward(input, output);

  const double sumModuleA = arma::accu(
      moduleA->Parameters().submat(
      100, 0, moduleA->Parameters().n_elem - 1, 0));
  const double sumModuleB = arma::accu(
      moduleB->Parameters().submat(
      100, 0, moduleB->Parameters().n_elem - 1, 0));
  REQUIRE(sumModuleA + sumModuleB ==
      Approx(arma::accu(output.col(0))).epsilon(1e-5));

  // Test the Backward function.
  error = arma::zeros(20, 1);
  delta.set_size(input.n_rows, input.n_cols);
  module.Backward(input, error, delta);
  REQUIRE(arma::accu(delta) == 0);
}

/**
 * Test to check Concat layer along different axes.
 */
TEST_CASE("ConcatAlongAxisTest", "[ANNLayerTest]")
{
  arma::mat output, input, error, outputA, outputB;
  size_t inputWidth = 4, inputHeight = 4, inputChannel = 2;
  size_t outputWidth, outputHeight, outputChannel = 2;
  size_t kW = 3, kH = 3;
  size_t batch = 1;

  // Using Convolution<> layer as inout to Concat<> layer.
  // Compute the output shape of convolution layer.
  outputWidth  = (inputWidth - kW) + 1;
  outputHeight = (inputHeight - kH) + 1;

  input = arma::ones(inputWidth * inputHeight * inputChannel, batch);

  Convolution* moduleA = new Convolution(outputChannel, kW, kH, 1, 1, 0, 0);
  Convolution* moduleB = new Convolution(outputChannel, kW, kH, 1, 1, 0, 0);

  moduleA->InputDimensions() = std::vector<size_t>({ inputWidth, inputHeight });
  moduleA->ComputeOutputDimensions();
  arma::mat weightsA(moduleA->WeightSize(), 1);
  moduleA->SetWeights((double*) weightsA.memptr());
  moduleA->Parameters().randu();

  moduleB->InputDimensions() = std::vector<size_t>({ inputWidth, inputHeight });
  moduleB->ComputeOutputDimensions();
  arma::mat weightsB(moduleB->WeightSize(), 1);
  moduleB->SetWeights((double*) weightsB.memptr());
  moduleB->Parameters().randu();

  // Compute output of each layer.
  outputA.set_size(moduleA->OutputSize(), 1);
  outputB.set_size(moduleB->OutputSize(), 1);
  moduleA->Forward(input, outputA);
  moduleB->Forward(input, outputB);

  arma::cube A(outputA.memptr(), outputWidth, outputHeight, outputChannel);
  arma::cube B(outputB.memptr(), outputWidth, outputHeight, outputChannel);

  error = arma::ones(outputWidth * outputHeight * outputChannel * 2, 1);

  for (size_t axis = 0; axis < 3; ++axis)
  {
    size_t x = 1, y = 1, z = 1;
    arma::cube calculatedOut;
    if (axis == 0)
    {
      calculatedOut.set_size(2 * outputWidth, outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
        arma::mat aMat = A.slice(i);
        arma::mat bMat = B.slice(i);
        calculatedOut.slice(i) = arma::join_cols(aMat, bMat);
      }
      x = 2;
    }
    if (axis == 1)
    {
      calculatedOut.set_size(outputWidth, 2 * outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
        arma::mat aMat = A.slice(i);
        arma::mat bMat = B.slice(i);
        calculatedOut.slice(i) = arma::join_rows(aMat, bMat);
      }
      y = 2;
    }
    if (axis == 2)
    {
      calculatedOut = arma::join_slices(A, B);
      z = 2;
    }

    // Compute output of Concat<> layer.
    Concat module(axis);
    module.Add(moduleA);
    module.Add(moduleB);
    module.InputDimensions() = std::vector<size_t>({ inputWidth, inputHeight });
    module.ComputeOutputDimensions();
    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);
    arma::cube concatOut(output.memptr(), x * outputWidth,
        y * outputHeight, z * outputChannel);

    // Verify if the output reshaped to cubes are similar.
    CheckMatrices(concatOut, calculatedOut, 1e-12);

    // Ensure that the child layers don't get deleted when `module` is
    // deallocated.
    module.Network().clear();
  }

  delete moduleA;
  delete moduleB;
}

/**
 * Test that the function that can access the axis parameter of the
 * Concat layer works.
 */
TEST_CASE("ConcatLayerParametersTest", "[ANNLayerTest]")
{
  Concat layer(2);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.Axis() == 2);
}

/**
 * Concat layer numerical gradient test.
 */
TEST_CASE("GradientConcatLayerTest", "[ANNLayerTest]")
{
  // Concat function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(10);

      concat = new Concat();
      concat->Add<Linear>(5);
      concat->Add<Linear>(5);
      model->Add(concat);
      model->Add<Linear>(2);

      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    Concat* concat;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
