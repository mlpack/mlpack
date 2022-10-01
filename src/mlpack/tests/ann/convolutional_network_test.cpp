/**
 * @file tests/convolutional_network_test.cpp
 * @author Marcus Edel
 * @author Abhinav Moudgil
 *
 * Tests the convolutional neural network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../serialization.hpp"
#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckCopyFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels,
                       const size_t maxEpochs)
{
  ens::RMSProp opt(0.01, 1, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);
  FFN<> network2;
  network2 = *network1;
  delete network1;

  // Deallocating all of network1's memory, so that network2 does not use any
  // of that memory.
  arma::mat predictions2;
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

// network1 should be allocated with `new`, and trained on some data.
template<typename MatType = arma::mat, typename ModelType>
void CheckMoveFunction(ModelType* network1,
                       MatType& trainData,
                       MatType& trainLabels,
                       const size_t maxEpochs)
{
  ens::RMSProp opt(0.01, 1, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  network1->Train(trainData, trainLabels, opt);

  arma::mat predictions1;
  network1->Predict(trainData, predictions1);
  FFN<> network2(std::move(*network1));
  delete network1;

  // Deallocating all of network1's memory, so that network2 does not use any
  // of that memory.
  arma::mat predictions2;
  network2.Predict(trainData, predictions2);
  CheckMatrices(predictions1, predictions2);
}

/**
 * Build a trivial network with a single padding layer, and make sure it
 * successfully pads the input.
 */
TEST_CASE("PaddingTest", "[ConvolutionalNetworktest]")
{
  arma::mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");

  // Create the network.
  FFN<NegativeLogLikelihood, RandomInitialization> model;

  model.Add<Padding>(1, 2, 3, 4);

  // Now, pass the data through.
  arma::mat results;
  model.InputDimensions() = std::vector<size_t>({ 28, 28 });
  model.Forward(X, results);

  // Ensure that things are correctly padded.
  arma::cube reshapedResults(results.memptr(), 31, 35, results.n_cols, false,
      true);

  for (size_t i = 0; i < reshapedResults.n_slices; ++i)
  {
    // Check left.
    for (size_t j = 0; j < reshapedResults.n_cols; ++j)
      REQUIRE(reshapedResults(0, j, i) == 0.0);

    // Check top.
    for (size_t j = 0; j < 3; ++j)
      for (size_t k = 0; k < reshapedResults.n_rows; ++k)
        REQUIRE(reshapedResults(k, j, i) == 0.0);

    // Check bottom.
    for (size_t j = 31; j < reshapedResults.n_cols; ++j)
      for (size_t k = 0; k < reshapedResults.n_rows; ++k)
        REQUIRE(reshapedResults(k, j, i) == 0.0);

    // Check right.
    for (size_t j = 0; j < reshapedResults.n_cols; ++j)
      for (size_t k = 29; k < reshapedResults.n_rows; ++k)
        REQUIRE(reshapedResults(k, j, i) == 0.0);
  }
}

/**
 * Build a trivial network with a MaxPooling layer, and make sure it
 * successfully does the max-pool operation.
 */
TEST_CASE("MaxPoolingTest", "[ConvolutionalNetworkTest]")
{
  arma::mat X(8, 3);
  X.col(0) = arma::vec("1, 2, 3, 4, 5, 6, 7, 8");
  X.col(1) = arma::vec("5, 7, 6, 8, 4, 3, 1, 2");
  X.col(2) = arma::vec("3, 4, 1, -1, 5, 5, 5, 5");

  // Create the network.
  FFN<NegativeLogLikelihood, RandomInitialization> model;
  model.Add<MaxPooling>(2, 2);

  arma::mat results;
  model.InputDimensions() = std::vector<size_t>({ 2, 4 });
  model.Forward(X, results);

  REQUIRE(results.n_rows == 3);
  REQUIRE(results.n_cols == 3);
  REQUIRE(results(0, 0) == 4);
  REQUIRE(results(1, 0) == 6);
  REQUIRE(results(2, 0) == 8);
  REQUIRE(results(0, 1) == 8);
  REQUIRE(results(1, 1) == 8);
  REQUIRE(results(2, 1) == 4);
  REQUIRE(results(0, 2) == 4);
  REQUIRE(results(1, 2) == 5);
  REQUIRE(results(2, 2) == 5);
}

/**
 * Train the vanilla network on a larger dataset.
 */
TEST_CASE("VanillaNetworkTest", "[ConvolutionalNetworkTest]")
{
  arma::mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  arma::uword nPoints = X.n_cols;
  for (arma::uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  // Build the target matrix.
  arma::mat Y = arma::zeros<arma::mat>(1, nPoints);
  for (size_t i = 0; i < nPoints; ++i)
  {
    if (i < nPoints / 2)
    {
      // Assign label "0" to all samples with digit = 4
      Y(i) = 0;
    }
    else
    {
      // Assign label "1" to all samples with digit = 9
      Y(i) = 1;
    }
  }

  /*
   * Construct a convolutional neural network with a 28x28x1 input layer,
   * 24x24x8 convolution layer, 12x12x8 pooling layer, 8x8x12 convolution layer
   * and a 4x4x12 pooling layer which is fully connected with the output layer.
   * The network structure looks like:
   *
   * Input    Convolution  Pooling      Convolution  Pooling      Output
   * Layer    Layer        Layer        Layer        Layer        Layer
   *
   *          +---+        +---+        +---+        +---+
   *          | +---+      | +---+      | +---+      | +---+
   * +---+    | | +---+    | | +---+    | | +---+    | | +---+    +---+
   * |   |    | | |   |    | | |   |    | | |   |    | | |   |    |   |
   * |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> |   |
   * |   |      +-+   |      +-+   |      +-+   |      +-+   |    |   |
   * +---+        +---+        +---+        +---+        +---+    +---+
   */
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights. If this works 1 of 5 times, I'm fine
  // with that. All I want to know is that the network is able to escape from
  // local minima and to solve the task.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    FFN<NegativeLogLikelihood, RandomInitialization> model;

    model.Add<Convolution>(8, 5, 5, 1, 1, 0, 0);
    model.Add<ReLU>();
    model.Add<MaxPooling>(2, 2);
    model.Add<Convolution>(12, 2, 2);
    model.Add<ReLU>();
    model.Add<MaxPooling>(2, 2);
    model.Add<Linear>(20);
    model.Add<ReLU>();
    model.Add<Linear>(10);
    model.Add<ReLU>();
    model.Add<Linear>(2);
    model.Add<LogSoftMax>();

    model.InputDimensions() = std::vector<size_t>({ 28, 28 });

    // Train for only 8 epochs.
    ens::RMSProp opt(0.001, 1, 0.88, 1e-8, 8 * nPoints, -1);

    double objVal = model.Train(X, Y, opt);

    // Test that objective value returned by FFN::Train() is finite.
    REQUIRE(std::isfinite(objVal) == true);

    arma::mat predictionTemp;
    model.Predict(X, predictionTemp);
    arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

    for (size_t i = 0; i < predictionTemp.n_cols; ++i)
    {
      prediction(i) = arma::as_scalar(arma::find(
            arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1));
    }

    size_t correct = arma::accu(prediction == Y);
    double classificationError = 1 - double(correct) / X.n_cols;
    if (classificationError <= 0.25)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

TEST_CASE("VanillaNetworkBatchSizeTest", "[ConvolutionalNetworkTest]")
{
  FFN<NegativeLogLikelihood, RandomInitialization> model;

  model.Add<Convolution>(8, 5, 5, 1, 1, 0, 0);
  model.Add<ReLU>();
  model.Add<MaxPooling>(2, 2);
  model.Add<Convolution>(12, 2, 2);
  model.Add<ReLU>();
  model.Add<MaxPooling>(2, 2);
  model.Add<Linear>(20);
  model.Add<ReLU>();
  model.Add<Linear>(10);
  model.Add<ReLU>();
  model.Add<Linear>(2);
  model.Add<LogSoftMax>();

  model.InputDimensions() = std::vector<size_t>({ 28, 28 });

  arma::mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  arma::uword nPoints = X.n_cols;
  for (arma::uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  // Build the target matrix.
  arma::mat Y = arma::zeros<arma::mat>(1, nPoints);
  for (size_t i = 0; i < nPoints; ++i)
  {
    if (i < nPoints / 2)
    {
      // Assign label "1" to all samples with digit = 4
      Y(i) = 1;
    }
    else
    {
      // Assign label "0" to all samples with digit = 9
      Y(i) = 0;
    }
  }

  // Perform one epoch of training to get the weights to somewhere reasonable.
  ens::RMSProp opt(0.001, 1, 0.88, 1e-8, nPoints, -1);
  model.Train(X, Y, opt);

  size_t trials = 7;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    const size_t batchSize = std::pow(2.0, (double) trial + 1.0);

    // Check the forward pass, and then call EvaluateWithGradient() to compute
    // the gradient.
    arma::mat results;
    arma::mat batchData = X.cols(0, batchSize - 1);
    arma::mat batchResponses = Y.cols(0, batchSize - 1);
    model.ResetData(std::move(batchData), std::move(batchResponses));
    model.Forward(X.cols(0, batchSize - 1), results);

    arma::mat gradient(1, model.WeightSize());
    const double obj = model.EvaluateWithGradient(model.Parameters(), gradient);

    REQUIRE(results.n_cols == batchSize);

    // Now compute results with a batch size of 1.
    arma::mat singleResults(results.n_rows, results.n_cols);
    arma::mat singleGradient(gradient.n_rows, gradient.n_cols, arma::fill::zeros);
    double singleObj = 0.0;

    for (size_t i = 0; i < batchSize; ++i)
    {
      arma::mat tmpResult;
      arma::mat singleData = X.cols(i, i);
      arma::mat singleResponses = Y.cols(i, i);
      model.ResetData(std::move(singleData), std::move(singleResponses));
      model.Forward(X.cols(i, i), tmpResult);
      REQUIRE(tmpResult.n_cols == 1);
      singleResults.col(i) = tmpResult;

      arma::mat tmpGradient(1, model.WeightSize());
      singleObj += model.EvaluateWithGradient(model.Parameters(), tmpGradient);

      singleGradient += tmpGradient;
    }

    // Check the forward pass results.
    CheckMatrices(results, singleResults);

    // Now, check EvaluateWithGradient()'s results.
    REQUIRE(obj == Approx(singleObj));
    CheckMatrices(gradient, singleGradient);
  }
}

/**
 * Train the vanilla network on a larger dataset.
 */
TEST_CASE("CheckCopyVanillaNetworkTest", "[ConvolutionalNetworkTest]")
{
  arma::mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  arma::uword nPoints = X.n_cols;
  for (arma::uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  // Build the target matrix.
  arma::mat Y = arma::zeros<arma::mat>(1, nPoints);
  for (size_t i = 0; i < nPoints; ++i)
  {
    if (i < nPoints / 2)
    {
      // Assign label "0" to all samples with digit = 4
      Y(i) = 0;
    }
    else
    {
      // Assign label "1" to all samples with digit = 9
      Y(i) = 1;
    }
  }

  /*
   * Construct a convolutional neural network with a 28x28x1 input layer,
   * 24x24x8 convolution layer, 12x12x8 pooling layer, 8x8x12 convolution layer
   * and a 4x4x12 pooling layer which is fully connected with the output layer.
   * The network structure looks like:
   *
   * Input    Convolution  Pooling      Convolution  Pooling      Output
   * Layer    Layer        Layer        Layer        Layer        Layer
   *
   *          +---+        +---+        +---+        +---+
   *          | +---+      | +---+      | +---+      | +---+
   * +---+    | | +---+    | | +---+    | | +---+    | | +---+    +---+
   * |   |    | | |   |    | | |   |    | | |   |    | | |   |    |   |
   * |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> |   |
   * |   |      +-+   |      +-+   |      +-+   |      +-+   |    |   |
   * +---+        +---+        +---+        +---+        +---+    +---+
   */
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights. If this works 1 of 5 times, I'm fine
  // with that. All I want to know is that the network is able to escape from
  // local minima and to solve the task.
  FFN<NegativeLogLikelihood, RandomInitialization> *model =
      new FFN<NegativeLogLikelihood, RandomInitialization>;

  model->Add<Convolution>(8, 5, 5, 1, 1, 0, 0);
  model->Add<ReLU>();
  model->Add<MaxPooling>(2, 2);
  model->Add<Convolution>(12, 2, 2);
  model->Add<ReLU>();
  model->Add<MaxPooling>(2, 2);
  model->Add<Linear>(20);
  model->Add<ReLU>();
  model->Add<Linear>(10);
  model->Add<ReLU>();
  model->Add<Linear>(2);
  model->Add<LogSoftMax>();
  model->InputDimensions() = std::vector<size_t>({ 28, 28 });

  FFN<NegativeLogLikelihood, RandomInitialization> *model1 =
      new FFN<NegativeLogLikelihood, RandomInitialization>;

  model1->Add<Convolution>(8, 5, 5, 1, 1, 0, 0);
  model1->Add<ReLU>();
  model1->Add<MaxPooling>(2, 2);
  model1->Add<Convolution>(12, 2, 2);
  model1->Add<ReLU>();
  model1->Add<MaxPooling>(2, 2);
  model1->Add<Linear>(20);
  model1->Add<ReLU>();
  model1->Add<Linear>(10);
  model1->Add<ReLU>();
  model1->Add<Linear>(2);
  model1->Add<LogSoftMax>();
  model1->InputDimensions() = std::vector<size_t>({ 28, 28 });

  // Check whether copy constructor is working or not.
  CheckCopyFunction<>(model, X, Y, 8);

  // Check whether move constructor is working or not.
  CheckMoveFunction<>(model1, X, Y, 8);
}

TEST_CASE("Issue2986", "[ConvolutionalNetworkTest]")
{
  // Ensure that the code snippet in issue #2986 succeeds without any issues.
  arma::mat input, output, delta;
  input.ones(36, 1);

  // Note that the stride here is 2, not 1.
  Convolution c(1, 3, 3, 2, 2, 0, 0);

  // Set up the layer without an enclosing FFN.
  c.InputDimensions() = std::vector<size_t>({ 6, 6 });
  c.ComputeOutputDimensions();
  arma::mat weights(c.WeightSize(), 1, arma::fill::randu);
  c.SetWeights(weights.memptr());

  output.set_size(c.OutputSize(), 1);
  delta.set_size(input.size());

  REQUIRE_NOTHROW(c.Forward(input, output));
  REQUIRE_NOTHROW(c.Backward(input, output, delta));

  // Now test with a stride of 3.
  c = Convolution(1, 3, 3, 3, 3, 0, 0);

  // Set up the layer without an enclosing FFN.
  c.InputDimensions() = std::vector<size_t>({ 6, 6 });
  c.ComputeOutputDimensions();
  weights.set_size(c.WeightSize(), 1);
  weights.randu();
  c.SetWeights(weights.memptr());

  output.set_size(c.OutputSize(), 1);
  delta.set_size(input.size());

  REQUIRE_NOTHROW(c.Forward(input, output));
  REQUIRE_NOTHROW(c.Backward(input, output, delta));

  // Now test with different strides for height and width.
  c = Convolution(1, 3, 3, 2, 3, 0, 0);

  // Set up the layer without an enclosing FFN.
  c.InputDimensions() = std::vector<size_t>({ 6, 6 });
  c.ComputeOutputDimensions();
  weights.set_size(c.WeightSize(), 1);
  weights.randu();
  c.SetWeights(weights.memptr());

  output.set_size(c.OutputSize(), 1);
  delta.set_size(input.size());

  REQUIRE_NOTHROW(c.Forward(input, output));
  REQUIRE_NOTHROW(c.Backward(input, output, delta));
}

// Test that the Convolution layer gives reasonable output when a non-zero
// padding size is used.
TEST_CASE("CustomPaddingTest", "[ConvolutionalNetworkTest]")
{
  arma::mat input, output, delta, weights;
  input.ones(36, 1);

  Convolution c = Convolution(1, 3, 3, 1, 1, { 1, 2 }, { 3, 4 }, "none");

  c.InputDimensions() = std::vector<size_t>({ 6, 6 });
  c.ComputeOutputDimensions();

  // First, check that the output dimensions are reasonable.
  REQUIRE(c.OutputDimensions().size() == 3);
  REQUIRE(c.OutputDimensions()[0] == 7);
  REQUIRE(c.OutputDimensions()[1] == 11);
  REQUIRE(c.OutputDimensions()[2] == 1);

  weights.set_size(c.WeightSize(), 1);
  weights.ones();
  c.SetWeights(weights.memptr());

  // Now make sure that the forward pass returns the correct output.
  output.set_size(c.OutputSize(), 1);
  REQUIRE_NOTHROW(c.Forward(input, output));
  REQUIRE(output.n_rows == c.OutputSize());
  REQUIRE(output.n_cols == 1);
  // The lower right corner's convolution entry should only touch one input
  // value (and everything else padding).
  REQUIRE(output(output.n_rows - 1, 0) == 1.0);

  delta.set_size(input.size());
  REQUIRE_NOTHROW(c.Backward(input, output, delta));
}
