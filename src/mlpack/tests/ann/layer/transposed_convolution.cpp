/**
 * @file tests/ann/layer/transposed_convolution.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 * @author Ranjodh Singh
 *
 * Test the TransposedConvolution layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/transposed_convolution.hpp>

#include "../../catch.hpp"

using namespace mlpack;


//! Calculates mean loss over batches.
template<typename NetworkType,
         typename DataType>
double MeanTestLoss(NetworkType& model, DataType& testSet, size_t batchSize)
{
  double loss = 0;
  size_t nofPoints = testSet.n_cols;

  size_t i;
  for (i = 0; i < (size_t) nofPoints / batchSize; i++)
  {
    loss += model.Evaluate(
        testSet.cols(batchSize * i, batchSize * (i + 1) - 1),
        testSet.cols(batchSize * i, batchSize * (i + 1) - 1));
  }

  if (nofPoints % batchSize != 0)
  {
    loss += model.Evaluate(testSet.cols(batchSize * i, nofPoints - 1),
                           testSet.cols(batchSize * i, nofPoints - 1));
    loss /= nofPoints / batchSize + 1;
  }
  else
  {
    loss /= nofPoints / batchSize;
  }

  return loss;
}

/**
 * Test that the functions that can modify and access the parameters of the
 * TransposedConvolution layer work.
 */
TEST_CASE("TransposedConvolutionParametersTest", "[ANNLayerTest]")
{
  // Parameter order: maps, kW, kH, dW, dH, padW, padH, paddingType.
  TransposedConvolution layer1(2, 3, 4, 5, 6, std::tuple<size_t, size_t>(7, 8),
      std::tuple<size_t, size_t>(9, 10), 0, 0, "none");
  TransposedConvolution layer2(3, 4, 5, 6, 7, std::tuple<size_t, size_t>(8, 9),
      std::tuple<size_t, size_t>(10, 11), 0, 0, "none");

  // Make sure we can get the parameters successfully.
  REQUIRE(layer1.KernelWidth() == 3);
  REQUIRE(layer1.KernelHeight() == 4);
  REQUIRE(layer1.StrideWidth() == 5);
  REQUIRE(layer1.StrideHeight() == 6);
  REQUIRE(layer1.PadWLeft() == 7);
  REQUIRE(layer1.PadWRight() == 8);
  REQUIRE(layer1.PadHTop() == 9);
  REQUIRE(layer1.PadHBottom() == 10);

  // Now modify the parameters to match the second layer.
  layer1.KernelWidth() = 4;
  layer1.KernelHeight() = 5;
  layer1.StrideWidth() = 6;
  layer1.StrideHeight() = 7;
  layer1.PadWLeft() = 8;
  layer1.PadWRight() = 9;
  layer1.PadHTop() = 10;
  layer1.PadHBottom() = 11;

  // Now ensure all results are the same.
  REQUIRE(layer1.KernelWidth() == layer2.KernelWidth());
  REQUIRE(layer1.KernelHeight() == layer2.KernelHeight());
  REQUIRE(layer1.StrideWidth() == layer2.StrideWidth());
  REQUIRE(layer1.StrideHeight() == layer2.StrideHeight());
  REQUIRE(layer1.PadWLeft() == layer2.PadWLeft());
  REQUIRE(layer1.PadWRight() == layer2.PadWRight());
  REQUIRE(layer1.PadHTop() == layer2.PadHTop());
  REQUIRE(layer1.PadHBottom() == layer2.PadHBottom());
}

/**
 * Test the functions that set weights of TransposedConvolution layer.
 */
TEST_CASE("TransposedConvolutionWeightInitializationTest", "[ANNLayerTest]")
{
  size_t maps = 3;
  size_t inMaps = 3;
  size_t kW = 4, kH = 4;
  TransposedConvolution module = TransposedConvolution(maps,
      kW, kH, 1, 1, 1, 1);
  module.InputDimensions() = std::vector<size_t>({5, 5, inMaps});
  module.ComputeOutputDimensions();

  const size_t biasSize = maps;
  const size_t weightSize = kW * kH * inMaps * maps;
  arma::mat weights = arma::randn(weightSize + biasSize);
  module.SetWeights(weights);

  arma::mat bias;
  arma::cube weight;
  MakeAlias(bias, weights, maps, 1, weightSize);
  MakeAlias(weight, weights, kW, kH, maps * inMaps);

  REQUIRE(arma::approx_equal(
    module.Bias(), bias, "absdiff", 0.0));
  REQUIRE(arma::approx_equal(
    module.Weight(), weight, "absdiff", 0.0));
  REQUIRE(arma::approx_equal(
    module.Parameters(), weights, "absdiff", 0.0));
}

/**
 * Test the functions that compute dimensions for TransposedConvolution layer.
 */
TEST_CASE("TransposedConvolutionDimensionsTest", "[ANNLayerTest]")
{
  struct Config
  {
    std::string paddingType;
    size_t inW;
    size_t inH;
    size_t kW;
    size_t kH;
    size_t dW;
    size_t dH;
    size_t pWLeft;
    size_t pWRight;
    size_t pHTop;
    size_t pHBottom;
    size_t outputPadW;
    size_t outputPadH;
    size_t expectedOutputWidth;
    size_t expectedOutputHeight;
  };

  // These tests cover different combinations of kernel, stride, and
  // padding dimensions, including equal and unequal width/height, odd and
  // even values, as well as cases where width > height or height > width.
  const std::vector<Config> configs = {
    // VALID (pads always treated as zero)
    { "valid",  7,  7, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0,  9,  9 },
    { "valid",  7,  8, 4, 2, 1, 2, 0, 0, 0, 0, 0, 0, 10, 16 },
    { "valid",  8,  7, 5, 1, 2, 1, 0, 0, 0, 0, 0, 0, 19,  7 },
    { "valid",  8,  8, 2, 5, 3, 3, 0, 0, 0, 0, 0, 0, 23, 26 },
    { "valid",  6,  9, 4, 4, 2, 1, 0, 0, 0, 0, 0, 0, 14, 12 },
    { "valid",  9,  6, 3, 5, 1, 3, 0, 0, 0, 0, 0, 0, 11, 20 },
    { "valid", 10, 10, 5, 2, 3, 1, 0, 0, 0, 0, 0, 0, 32, 11 },
    { "valid",  5,  5, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0,  6, 11 },
    { "valid",  5,  6, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 11, 14 },

    // NONE (explicit pads)
    { "none",  7,  7, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0,  7,  7 },
    { "none",  8,  8, 4, 2, 2, 2, 1, 2, 2, 1, 0, 0, 15, 13 },
    { "none",  7,  8, 5, 3, 1, 2, 0, 1, 1, 0, 0, 0, 10, 16 },
    { "none",  8,  7, 2, 5, 3, 1, 2, 0, 0, 2, 0, 0, 21,  9 },
    { "none",  6,  6, 3, 4, 2, 3, 1, 1, 2, 0, 0, 0, 11, 17 },
    { "none",  9,  5, 4, 2, 1, 3, 0, 3, 1, 1, 0, 0,  9, 12 },
    { "none",  5,  9, 5, 5, 2, 2, 2, 2, 2, 2, 0, 0,  9, 17 },
    { "none", 10, 10, 3, 3, 3, 3, 1, 4, 0, 3, 0, 0, 25, 27 },

    // SAME (output spatial == input spatial, stride == 1)
    { "same",  7,  7, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0,  7,  7 },
    { "same",  8,  8, 4, 2, 1, 1, 1, 2, 2, 1, 0, 0,  8,  8 },
    { "same",  7,  8, 5, 3, 1, 1, 0, 1, 1, 0, 0, 0,  7,  8 },
    { "same",  8,  7, 2, 5, 1, 1, 2, 0, 0, 2, 0, 0,  8,  7 },
    { "same",  6,  6, 3, 4, 1, 1, 1, 1, 2, 0, 0, 0,  6,  6 },
    { "same",  9,  5, 4, 2, 1, 1, 0, 3, 1, 1, 0, 0,  9,  5 },
    { "same",  5,  9, 5, 5, 1, 1, 2, 2, 2, 2, 0, 0,  5,  9 },
    { "same", 10, 10, 3, 3, 1, 1, 1, 4, 0, 3, 0, 0, 10, 10 },

    // Output Padding
    { "valid", 9,  6, 3, 5, 1, 3, 0, 0, 0, 0, 0, 1, 11, 21 },
    { "none",  5,  9, 5, 5, 2, 2, 2, 2, 2, 2, 1, 1, 10, 18 },
    { "same",  7,  7, 3, 3, 1, 1, 0, 0, 0, 0, 1, 0,  8,  7 },
  };

  const size_t inMaps = 1, maps = 2;
  for (size_t i = 0; i < configs.size(); ++i)
  {
    const Config& c = configs[i];
    const std::string sectionName = "Case - " + std::to_string(i);
    SECTION(sectionName)
    {
      TransposedConvolution module(
        maps,
        c.kW,
        c.kH,
        c.dW,
        c.dH,
        std::make_tuple(c.pWLeft, c.pWRight),
        std::make_tuple(c.pHTop, c.pHBottom),
        c.outputPadW,
        c.outputPadH,
        c.paddingType);

      module.InputDimensions() = { c.inW, c.inH, inMaps };
      module.ComputeOutputDimensions();

      // WeightSize = inMaps * maps * kernelWidth * kernelHeight + maps (bias).
      REQUIRE(module.WeightSize() == inMaps * maps * c.kW * c.kH + maps);
      // OutputSize = (outputWidth * outputHeight * maps).
      REQUIRE(module.OutputSize() == c.expectedOutputWidth *
          c.expectedOutputHeight * maps);

      REQUIRE(module.OutputDimensions() == std::vector<size_t>{
        c.expectedOutputWidth, c.expectedOutputHeight, maps});
    }
  }
}

/**
 * Test the forward, backward, and gradient functions
 * of the TransposedConvolution layer.
 */
TEST_CASE("TransposedConvolutionForwardBackwardGradientTest", "[ANNLayerTest]")
{
  struct Config
  {
    size_t maps;
    size_t kW;
    size_t kH;
    size_t dW;
    size_t dH;
    size_t pW;
    size_t pH;
    size_t oW;
    size_t oH;
    size_t inW;
    size_t inH;
    std::map<size_t, double> weightAssignments;

    double expectedOutputSum;
    double expectedDeltaSum;
    double expectedGradSum;
    size_t expectedTotalWeights; // Including bias
  };

  // The expected values for Output, Delta
  // and Gradient sums were calculated using pytorch.
  // https://gist.github.com/ranjodhsingh1729/48f28648187fd4eed7d30c95069808f7
  std::vector<Config> configs = {
      {
        1, 3, 3, 1, 1, 0, 0, 0, 0, 4, 4,
        {{0, 1.0}, {8, 2.0}},
        360.0, 720.0, 15915.0, 10
      },
      {
        1, 4, 4, 1, 1, 1, 1, 0, 0, 5, 5,
        {{0, 1.0}, {3, 1.0}, {6, 1.0}, {9, 1.0}, {12, 1.0}, {15, 2.0}},
        1512.0, 6504.0, 215350.0, 17
      },
      {
        1, 3, 3, 1, 1, 1, 1, 0, 0, 5, 5,
        {{1, 2.0}, {2, 4.0}, {3, 3.0}, {8, 1.0}},
        2370.0, 19154.0, 240789.0, 10
      },
      {
        1, 3, 3, 1, 1, 0, 0, 0, 0, 5, 5,
        {{2, 2.0}, {4, 4.0}, {6, 6.0}, {8, 8.0}},
        6000.0, 86208.0, 524352.0, 10
      },
      {
        1, 3, 3, 2, 2, 0, 0, 0, 0, 2, 2,
        {{2, 8.0}, {4, 6.0}, {6, 4.0}, {8, 2.0}},
        120.0, 960.0, 550.0, 10
      },
      {
        1, 3, 3, 2, 2, 1, 1, 0, 0, 3, 3,
        {{0, 8.0}, {3, 6.0}, {6, 2.0}, {8, 4.0}},
        410.0, 4444.0, 6684.0, 10
      },
      {
        1, 3, 3, 2, 2, 1, 1, 0, 0, 3, 3,
        {{0, 8.0}, {2, 6.0}, {4, 2.0}, {8, 4.0}},
        416.0, 6336.0, 7048.0, 10
      },
      {
        1, 3, 1, 1, 2, 0, 0, 0, 0, 4, 1,
        {{0, 1.0}, {1, 0.0}, {2, -1.0}},
        0.0, 6.0, 0.0, 4
      },
      {
        1, 3, 2, 1, 1, 0, 0, 0, 0, 5, 4,
        {{0, 1.0}, {2, 2.0}, {5, 3.0}},
        1140.0, 5339.0, 58272.0, 7
      },
      {
        1, 3, 2, 1, 1, 0, 1, 0, 0, 3, 4,
        {{0, 1.0}, {1, 3.0}, {4, 5.0}, {5, 7.0}},
        684.0, 8658.0, 20829.0, 7
      },
      {
        1, 2, 3, 2, 2, 0, 0, 0, 0, 3, 3,
        {{0, 1.0}, {3, 2.0}, {5, 3.0}},
        216.0, 504.0, 1840.0, 7
      },
      {
        1, 2, 3, 2, 1, 0, 1, 0, 0, 3, 4,
        {{0, 1.0}, {1, 2.0}, {2, 3.0}, {5, 4.0}},
        531.0, 2310.0, 9439.0, 7
      },
      {
        1, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1,
        {{0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 4.0}, {4, 5.0},
         {5, 6.0}, {6, 7.0}, {7, 8.0}, {8, 9.0}},
        0.0, 0.0, 0.0, 10
      },
      {
        1, 3, 1, 1, 2, 1, 0, 0, 0, 1, 4,
        {{0, 1.0}, {1, 2.0}, {2, 3.0}},
        12.0, 24.0, 40.0, 4
      },
      {
        1, 3, 3, 1, 1, 1, 1, 0, 0, 4, 4,
        {{9, 3.0}},
        48.0, 0.0, 2298.0, 10
      },
      {
        1, 3, 3, 2, 2, 1, 1, 1, 1, 3, 3,
        {{0, 8.0}, {2, 6.0}, {4, 2.0}, {8, 4.0}},
        606.0, 7732.0, 9376.0, 10
      },
  };

  for (size_t i = 0; i < configs.size(); ++i)
  {
    const Config& c = configs[i];
    std::string sectionName = "Case - " + std::to_string(i);
    SECTION(sectionName)
    {
      arma::mat input, output, weights, delta, grad;
      TransposedConvolution module(c.maps, c.kW, c.kH, c.dW, c.dH, c.pW, c.pH,
          c.oW, c.oH);

      size_t inputSize = c.inW * c.inH;
      input = arma::linspace<arma::colvec>(0, inputSize - 1, inputSize);

      module.InputDimensions() = {c.inW, c.inH};
      module.ComputeOutputDimensions();

      weights.set_size(module.WeightSize(), 1);
      weights.zeros();

      const size_t kernelSize = c.kW * c.kH;
      for (auto const& [index, value] : c.weightAssignments)
      {
        const size_t newIndex = index < kernelSize
                                ? kernelSize - index - 1
                                : index;
        weights[newIndex] = value;
      }
      module.SetWeights(weights);

      output.set_size(module.OutputSize(), 1);
      module.Forward(input, output);

      delta.set_size(arma::size(input));
      module.Backward(input, output, output, delta);

      grad.set_size(arma::size(weights));
      module.Gradient(input, output, grad);

      REQUIRE(accu(output) == c.expectedOutputSum);
      REQUIRE(accu(delta) == c.expectedDeltaSum);
      REQUIRE(accu(grad) == c.expectedGradSum);
      REQUIRE(weights.n_elem == c.expectedTotalWeights);
    }
  }
}

/**
 * Test the numerical gradient of the TransposedConvolution layer.
 */
TEST_CASE("TransposedConvolutionGradientTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::linspace<arma::colvec>(0, 35, 36)),
        target(arma::mat("1"))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>();
      model->ResetData(input, target);
      model->Add<TransposedConvolution>(1, 3, 3, 1, 1, 0, 0, 0, 0, "same");
      model->Add<LogSoftMax>();
      model->InputDimensions() = std::vector<size_t>({ 6, 6 });
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

    FFN<NegativeLogLikelihood, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) < 2e-4);
}

/**
 * Create a simple autoencoder with Convolution and Transposed Convolution
 * layers, train it on a subset of MNIST, and check the mean squared error.
 */
TEST_CASE("ConvTransConvAutoencoderTest", "[ANNLayerTest][long]")
{
  FFN<MeanSquaredError, XavierInitialization> autoencoder;
  autoencoder.Add<Convolution>(16, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<LeakyReLU>();
  autoencoder.Add<Convolution>(32, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<LeakyReLU>();
  autoencoder.Add<TransposedConvolution>(16, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<LeakyReLU>();
  autoencoder.Add<TransposedConvolution>(1, 2, 2, 2, 2, 0, 0);
  autoencoder.Add<Sigmoid>();
  autoencoder.InputDimensions() = std::vector<size_t>({28, 28});

  TextOptions opts;
  opts.Fatal() = true;
  opts.NoTranspose() = true;

  arma::mat data;
  Load("mnist_first250_training_4s_and_9s.csv", data, opts);
  data = (data - data.min()) / (data.max() - data.min());

  arma::mat trainData, testData;
  Split(data, trainData, testData, 0.2);

  ens::Adam optimizer;
  optimizer.StepSize() = 0.1;
  optimizer.BatchSize() = 16;
  optimizer.MaxIterations() = 2 * trainData.n_cols;
  autoencoder.Train(trainData, trainData, optimizer);

  REQUIRE(MeanTestLoss<>(autoencoder, testData, optimizer.BatchSize()) < 0.1);
}
