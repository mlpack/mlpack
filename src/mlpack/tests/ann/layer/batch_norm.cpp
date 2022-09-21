/**
 * @file tests/ann/layer/batch_norm.cpp
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
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * Tests the BatchNorm Layer, compares the layers parameters with
 * the values from another implementation.
 * Link to the implementation - http://cthorey.github.io./backpropagation/
 */
TEST_CASE("BatchNormTest", "[ANNLayerTest]")
{
  arma::mat output;
  arma::mat input = { { 5.1, 3.5, 1.4 },
                      { 4.9, 3.0, 1.4 },
                      { 4.7, 3.2, 1.3 } };

  input = input.t();

  // BatchNorm layer with average parameter set to true.
  BatchNorm module1;
  module1.Training() = true;
  module1.InputDimensions() = std::vector<size_t>({ 3, 3 });
  module1.ComputeOutputDimensions();
  arma::mat moduleParams(module1.WeightSize(), 1);
  module1.CustomInitialize(moduleParams, module1.WeightSize());
  module1.SetWeights((double*) moduleParams.memptr());

  // BatchNorm layer with average parameter set to false (using momentum).
  BatchNorm module2(2, 2, 1e-5, false);
  module2.Training() = true;
  module2.InputDimensions() = std::vector<size_t>({ 3, 3 });
  module2.ComputeOutputDimensions();
  arma::mat moduleParams2(module2.WeightSize(), 1);
  module2.CustomInitialize(moduleParams2, module2.WeightSize());
  module2.SetWeights((double*) moduleParams2.memptr());

  // Training Forward Pass Test.
  output.set_size(module1.OutputSize(), 1);
  input.reshape(9, 1);
  module1.Forward(input, output);

 // Value calculates using torch.nn.BatchNorm1d(momentum = None).
  arma::mat result;
  output.reshape(3, 3);
  result = { { 1.1658, 0.1100, -1.2758 },
            { 1.2579, -0.0699, -1.1880},
            { 1.1737, 0.0958, -1.2695 } };

  CheckMatrices(output, result.t(), 1e-1);

  output.set_size(module2.OutputSize(), 1);
  module2.Forward(input, output);
  output.reshape(3, 3);
  CheckMatrices(output, result.t(), 1e-1);
  result.clear();
  output.clear();

 // Values calculated using torch.nn.BatchNorm1d(momentum = None).
  output = module1.TrainingMean();
  result = arma::mat({ 3.33333333, 3.1, 3.06666666 }).t();

  CheckMatrices(output, result, 1e-1);

 // Values calculated using torch.nn.BatchNorm1d().
  output = module2.TrainingMean();
  result = arma::mat({ 0.3333, 0.3100, 0.3067 }).t();

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Values calculated using torch.nn.BatchNorm1d(momentum = None).
  output = module1.TrainingVariance();
  result = arma::mat({ 3.4433, 3.0700, 2.9033 }).t();

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Values calculated using torch.nn.BatchNorm1d().
  output = module2.TrainingVariance();
  result = arma::mat({ 1.2443, 1.2070, 1.1903 }).t();

  CheckMatrices(output, result, 1e-1);
  result.clear();

  // Deterministic Forward Pass test.
  module1.Training() = false;
  output.set_size(module1.OutputSize(), 1);
  module1.Forward(input, output);
  output.reshape(3, 3);

  // Values calculated using torch.nn.BatchNorm1d(momentum = None).
  result = { { 0.9521, 0.0898, -1.0419 },
            { 1.0273, -0.0571, -0.9702 },
            { 0.9586, 0.0783, -1.0368 } };

  CheckMatrices(output, result.t(), 1e-1);

  // Values calculated using torch.nn.BatchNorm1d().
  module2.Training() = false;
  output.set_size(module2.OutputSize(), 1);
  module2.Forward(input, output);
  output.reshape(3, 3);

  result = { { 4.2731, 2.8388, 0.9562 },
             { 4.1779, 2.4485, 0.9921 },
             { 4.0268, 2.6519, 0.9105 } };

  CheckMatrices(output, result.t(), 1e-1);
}

/**
 * BatchNorm layer numerical gradient test.
 */
TEST_CASE("GradientBatchNormTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randn(32, 2048)),
        target(arma::zeros(1, 2048))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(4);
      model->Add<BatchNorm>();
      model->Add<Linear>(2);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 2048);
      model->Gradient(model->Parameters(), 0, gradient, 2048);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  double gradient = CheckGradient(function);

  REQUIRE(gradient < 1e-1);
}

// General ANN serialization test.
template<typename LayerType>
void ANNLayerSerializationTest(LayerType& layer)
{
  arma::mat input(5, 100, arma::fill::randu);
  arma::mat output(5, 100, arma::fill::randu);

  FFN<> model;
  model.Add<Linear>(10);
  model.Add<LayerType>(layer);
  model.Add<ReLU>();
  model.Add<Linear>(output.n_rows);
  model.Add<LogSoftMax>();

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  model.Train(input, output, opt);

  arma::mat originalOutput;
  model.Predict(input, originalOutput);

  // Now serialize the model.
  FFN<> xmlModel, jsonModel, binaryModel;
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  // Ensure that predictions are the same.
  arma::mat modelOutput, xmlOutput, jsonOutput, binaryOutput;
  model.Predict(input, modelOutput);
  xmlModel.Predict(input, xmlOutput);
  jsonModel.Predict(input, jsonOutput);
  binaryModel.Predict(input, binaryOutput);

  CheckMatrices(originalOutput, modelOutput, 1e-5);
  CheckMatrices(originalOutput, xmlOutput, 1e-5);
  CheckMatrices(originalOutput, jsonOutput, 1e-5);
  CheckMatrices(originalOutput, binaryOutput, 1e-5);
}

/**
 * Simple serialization test for batch normalization layer.
 */
TEST_CASE("BatchNormSerializationTest", "[ANNLayerTest]")
{
  BatchNorm layer;
  ANNLayerSerializationTest(layer);
}

TEST_CASE("BatchNormWithMinBatchesTest", "[ANNLayerTest]")
{
  arma::mat input, output, result, runningMean, runningVar, delta;

  // The input test matrix is of the form 3 x 2 x 4 x 1 where
  // number of images are 3 and number of feature maps are 2.
  input = { { 1, 446, 42 },
            { 2, 16, 63 },
            { 3, 13, 63 },
            { 4, 21, 21 },
            { 1, 13, 11 },
            { 32, 45, 42 },
            { 22, 16, 63 },
            { 32, 13, 42 } };

  // Output calculated using torch.nn.BatchNorm2d().
  result = { { -0.4786, 3.2634, -0.1338 },
             { -0.4702, -0.3525, 0.0427 },
             { -0.4618, -0.3777, 0.0427 },
             { -0.4534, -0.3104, -0.3104 },
             { -1.5429, -0.8486, -0.9643 },
             { 0.2507, 1.0029, 0.8293 },
             { -0.3279, -0.675, 2.0443 },
             { 0.2507 , -0.8486 , 0.8293 } };

  // Check correctness of batch normalization.
  BatchNorm module1(2, 2, 1e-5, false, 0.1);
  module1.Training() = true;
  module1.InputDimensions() = std::vector<size_t>({ 1, 4, 2 });
  module1.ComputeOutputDimensions();
  arma::mat moduleParams(module1.WeightSize(), 1);
  module1.CustomInitialize(moduleParams, module1.WeightSize());
  module1.SetWeights((double*) moduleParams.memptr());
  output.set_size(8, 3);
  module1.Forward(input, output);
  CheckMatrices(output, result, 1e-1);

  // Check values for running mean and running variance.
  // Calculated using torch.nn.BatchNorm2d().
  runningMean = arma::mat(2, 1);
  runningVar = arma::mat(2, 1);
  runningMean(0) = 5.7917;
  runningMean(1) = 2.76667;
  runningVar(0) = 1543.6545;
  runningVar(1) = 33.488;

  CheckMatrices(runningMean, module1.TrainingMean(), 1e-3);
  CheckMatrices(runningVar, module1.TrainingVariance(), 1e-2);

  // Check correctness of layer when running mean and variance
  // are updated using cumulative average.
  BatchNorm module2;
  module2.Training() = true;
  module2.InputDimensions() = std::vector<size_t>({ 1, 4, 2 });
  module2.ComputeOutputDimensions();
  arma::mat moduleParams2(module2.WeightSize(), 1);
  module2.CustomInitialize(moduleParams2, module2.WeightSize());
  module2.SetWeights((double*) moduleParams2.memptr());
  output.set_size(8, 3);
  module2.Forward(input, output);
  CheckMatrices(output, result, 1e-1);

  // Check values for running mean and running variance.
  // Calculated using torch.nn.BatchNorm2d().
  runningMean(0) = 57.9167;
  runningMean(1) = 27.6667;
  runningVar(0) = 15427.5380;
  runningVar(1) = 325.8787;

  CheckMatrices(runningMean, module2.TrainingMean(), 1e-2);
  CheckMatrices(runningVar, module2.TrainingVariance(), 1e-2);

  // Check correctness when model is testing.
  arma::mat deterministicOutput;
  module1.Training() = false;
  deterministicOutput.set_size(8, 3);
  module1.Forward(input, deterministicOutput);

  result.clear();
  result = { { -0.12195, 11.20426, 0.92158 },
             { -0.0965, 0.259824, 1.4560 },
             { -0.071054, 0.183567, 1.45607 },
             { -0.045601, 0.3870852, 0.38708 },
             { -0.305288, 1.7683, 1.4227 },
             { 5.05166, 7.29812, 6.7797 },
             { 3.323614, 2.2867, 10.4086 },
             { 5.05166, 1.7683, 6.7797 } };

  CheckMatrices(result, deterministicOutput, 1e-1);
}

/**
 * Batch Normalization layer numerical gradient test.
 */
TEST_CASE("GradientBatchNormWithConvolutionTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randn(16, 1024)),
        target(arma::zeros(1, 1024))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Convolution>(2, 3, 3);
      model->Add<BatchNorm>();
      model->Add<Linear>(2);
      model->Add<LogSoftMax>();

      model->InputDimensions() = std::vector<size_t>({ 4, 4, 1 });
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1024);
      model->Gradient(model->Parameters(), 0, gradient, 1024);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  double gradient = CheckGradient(function);

  REQUIRE(gradient < 1e-1);
}
