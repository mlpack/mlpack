/**
 * @file rmva_main.cpp
 * @author Marcus Edel
 *
 * Main executable for the Recurrent Model for Visual Attention.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "rmva.hpp"

#include <mlpack/methods/ann/layer/glimpse_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/reinforce_normal_layer.hpp>
#include <mlpack/methods/ann/layer/multiply_constant_layer.hpp>
#include <mlpack/methods/ann/layer/constant_layer.hpp>
#include <mlpack/methods/ann/layer/log_softmax_layer.hpp>
#include <mlpack/methods/ann/layer/hard_tanh_layer.hpp>

#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace std;

PROGRAM_INFO("Recurrent Model for Visual Attention",
    "This program trains the Recurrent Model for Visual Attention on the given "
    "labeled training set, or loads a model from the given model file, and then"
    " may use that trained model to classify the points in a given test set."
    "\n\n"
    "Labels are expected to be passed in separately as their own file "
    "(--labels_file).  If training is not desired, a pre-existing model can be "
    "loaded with the --input_model_file (-m) option."
    "\n\n"
    "If classifying a test set is desired, the test set should be in the file "
    "specified with the --test_file (-T) option, and the classifications will "
    "be saved to the file specified with the --output_file (-o) option.  If "
    "saving a trained model is desired, the --output_model_file (-M) option "
    "should be given.");

// Model loading/saving.
PARAM_STRING_IN("input_model_file", "File containing the Recurrent Model for "
    "Visual Attention.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained Recurrent Model for"
    " Visual Attention to.", "M");

// Training parameters.
PARAM_MATRIX_IN("training", "Matrix containing the training set.", "t");
PARAM_MATRIX_IN("labels", "Matrix containing labels for the training set.",
    "l");

PARAM_STRING_IN("optimizer", "Optimizer to use; 'sgd', 'minibatch-sgd', or "
    "'lbfgs'.", "O", "minibatch-sgd");

PARAM_INT_IN("max_iterations", "Maximum number of iterations for SGD or RMSProp"
    " (0 indicates no limit).", "n", 500000);
PARAM_DOUBLE_IN("tolerance", "Maximum tolerance for termination of SGD or "
    "RMSProp.", "e", 1e-7);

PARAM_DOUBLE_IN("step_size", "Step size for stochastic gradient descent "
    "(alpha),", "a", 0.01);
PARAM_FLAG("linear_scan", "Don't shuffle the order in which data points are "
    "visited for SGD or mini-batch SGD.", "L");
PARAM_INT_IN("batch_size", "Batch size for mini-batch SGD.", "b", 20);

PARAM_INT_IN("rho", "Number of steps for the back-propagate through time.", "r",
    7);

PARAM_INT_IN("classes", "The number of classes.", "c", 10);

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

// Test parameters.
PARAM_MATRIX_IN("test", "Matrix containing the test set.", "T");
PARAM_MATRIX_OUT("output", "The matrix in which the predicted labels for the "
    "test set will be written.", "o");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Check input parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Cannot specify both --training_file (-t) and "
       << "--input_model_file (-m)!" << endl;

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Neither --training_file (-t) nor --input_model_file (-m) are"
        << " specified!" << endl;

  if (!CLI::HasParam("training") && CLI::HasParam("labels"))
    Log::Warn << "--labels_file (-l) ignored because --training_file (-t) is "
        << "not specified." << endl;

  if (!CLI::HasParam("output") && !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither --output_file (-o) nor --output_model_file (-M) "
        << "specified; no output will be saved!" << endl;

  if (CLI::HasParam("output") && !CLI::HasParam("test"))
    Log::Warn << "--output_file (-o) ignored because no test file specified "
        << "with --test_file (-T)." << endl;

  if (!CLI::HasParam("output") && CLI::HasParam("test"))
    Log::Warn << "--test_file (-T) specified, but classification results will "
        << "not be saved because --output_file (-o) is not specified." << endl;

  const string optimizerType = CLI::GetParam<string>("optimizer");

  if ((optimizerType != "sgd") && (optimizerType != "lbfgs") &&
      (optimizerType != "minibatch-sgd"))
  {
    Log::Fatal << "Optimizer type '" << optimizerType << "' unknown; must be "
        << "'sgd', 'minibatch-sgd', or 'lbfgs'!" << endl;
  }

  const double stepSize = CLI::GetParam<double>("step_size");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const bool shuffle = !CLI::HasParam("linear_scan");
  const size_t batchSize = (size_t) CLI::GetParam<int>("batch_size");
  const size_t rho = (size_t) CLI::GetParam<int>("rho");
  const size_t numClasses = (size_t) CLI::GetParam<int>("classes");

  const size_t hiddenSize = 256;
  const double unitPixels = 13;
  const double locatorStd = 0.11;
  const size_t imageSize = 28;
  const size_t locatorHiddenSize = 128;
  const size_t glimpsePatchSize = 8;
  const size_t glimpseDepth = 1;
  const size_t glimpseScale = 2;
  const size_t glimpseHiddenSize = 128;
  const size_t imageHiddenSize = 256;


  // Locator network.
  LinearMappingLayer<> linearLayer0(hiddenSize, 2);
  BiasLayer<> biasLayer0(2, 1);
  HardTanHLayer<> hardTanhLayer0;
  ReinforceNormalLayer<> reinforceNormalLayer0(2 * locatorStd);
  HardTanHLayer<> hardTanhLayer1;
  MultiplyConstantLayer<> multiplyConstantLayer0(2 * unitPixels / imageSize);
  auto locator = std::tie(linearLayer0, biasLayer0, hardTanhLayer0,
      reinforceNormalLayer0, hardTanhLayer1, multiplyConstantLayer0);

  // Location sensor network.
  LinearLayer<> linearLayer1(2, locatorHiddenSize);
  BiasLayer<> biasLayer1(locatorHiddenSize, 1);
  ReLULayer<> rectifierLayer0;
  auto locationSensor = std::tie(linearLayer1, biasLayer1, rectifierLayer0);

  // Glimpse sensor network.
  GlimpseLayer<> glimpseLayer0(1, glimpsePatchSize, glimpseDepth, glimpseScale);
  LinearMappingLayer<> linearLayer2(64, glimpseHiddenSize);
  BiasLayer<> biasLayer2(glimpseHiddenSize, 1);
  ReLULayer<> rectifierLayer1;
  auto glimpseSensor = std::tie(glimpseLayer0, linearLayer2, biasLayer2,
      rectifierLayer1);

  // Glimpse network.
  LinearLayer<> linearLayer3(glimpseHiddenSize + locatorHiddenSize,
      imageHiddenSize);
  BiasLayer<> biasLayer3(imageHiddenSize, 1);
  ReLULayer<> rectifierLayer2;
  LinearLayer<> linearLayer4(imageHiddenSize, hiddenSize);
  BiasLayer<> biasLayer4(hiddenSize, 1);
  auto glimpse = std::tie(linearLayer3, biasLayer3, rectifierLayer2,
      linearLayer4, biasLayer4);

  // Feedback network.
  LinearLayer<> recurrentLayer0(imageHiddenSize, hiddenSize);
  BiasLayer<> recurrentLayerBias0(hiddenSize, 1);
  auto feedback = std::tie(recurrentLayer0, recurrentLayerBias0);

  // Start network.
  AdditionLayer<> startLayer0(hiddenSize, 1);
  auto start = std::tie(startLayer0);

  // Transfer network.
  ReLULayer<> rectifierLayer3;
  auto transfer = std::tie(rectifierLayer3);

  // Classifier network.
  LinearLayer<> linearLayer5(hiddenSize, numClasses);
  BiasLayer<> biasLayer6(numClasses, 1);
  LogSoftmaxLayer<> logSoftmaxLayer0;
  auto classifier = std::tie(linearLayer5, biasLayer6, logSoftmaxLayer0);

  // Reward predictor network.
  ConstantLayer<> constantLayer0(1, 1);
  AdditionLayer<> additionLayer0(1, 1);
  auto rewardPredictor = std::tie(constantLayer0, additionLayer0);

  // Recurrent Model for Visual Attention.
  RecurrentNeuralAttention<decltype(locator),
                           decltype(locationSensor),
                           decltype(glimpseSensor),
                           decltype(glimpse),
                           decltype(start),
                           decltype(feedback),
                           decltype(transfer),
                           decltype(classifier),
                           decltype(rewardPredictor),
                           RandomInitialization>
    net(locator, locationSensor, glimpseSensor, glimpse, start, feedback,
        transfer, classifier, rewardPredictor, rho);

  // Either we have to train a model, or load a model.
  if (CLI::HasParam("training"))
  {
    arma::mat trainingData = std::move(CLI::GetParam<arma::mat>("training"));

    arma::mat labels;

    // Did the user pass in labels?
    if (CLI::HasParam("labels"))
    {
      // Load labels.
      labels = std::move(CLI::GetParam<arma::mat>("labels"));

      // Do the labels need to be transposed?
      if (labels.n_cols == 1)
        labels = labels.t();
    }

    // Now run the optimization.
    if (optimizerType == "sgd")
    {
      SGD<decltype(net)> opt(net);
      opt.StepSize() = stepSize;
      opt.MaxIterations() = maxIterations;
      opt.Tolerance() = tolerance;
      opt.Shuffle() = shuffle;

      Timer::Start("rmva_training");
      net.Train(trainingData, labels, opt);
      Timer::Stop("rmva_training");
    }
    else if (optimizerType == "minibatch-sgd")
    {
      MiniBatchSGD<decltype(net)> opt(net);
      opt.StepSize() = stepSize;
      opt.MaxIterations() = maxIterations;
      opt.Tolerance() = tolerance;
      opt.Shuffle() = shuffle;
      opt.BatchSize() = batchSize;

      Timer::Start("rmva_training");
      net.Train(trainingData, labels, opt);
      Timer::Stop("rmva_training");
    }
  }
  else
  {
    // Load the model from file.
    data::Load(CLI::GetParam<string>("input_model_file"), "rmva_model", net);
  }

  // Do we need to do testing?
  if (CLI::HasParam("test"))
  {
    arma::mat testingData = std::move(CLI::GetParam<arma::mat>("test"));

    // Time the running of the Naive Bayes Classifier.
    arma::mat results;
    Timer::Start("rmva_testing");
    net.Predict(testingData, results);
    Timer::Stop("rmva_testing");

    if (CLI::HasParam("output"))
      CLI::GetParam<arma::mat>("output") = std::move(results);
  }

  // Save the model, if requested.
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "rmva_model", net);
}
