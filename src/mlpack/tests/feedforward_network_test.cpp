/**
 * @file feedforward_network_test.cpp
 * @author Marcus Edel
 * @author Palash Ahuja
 *
 * Tests the feed forward network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"
#include "custom_layer.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(FeedForwardNetworkTest);

/**
 * Train and evaluate a model with the specified structure.
 */
template<typename MatType = arma::mat, typename ModelType>
void TestNetwork(ModelType& model,
                 MatType& trainData,
                 MatType& trainLabels,
                 MatType& testData,
                 MatType& testLabels,
                 const size_t maxEpochs,
                 const double classificationErrorThreshold)
{
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (int(arma::as_scalar(prediction.col(i))) ==
        int(arma::as_scalar(testLabels.col(i))))
    {
      error++;
    }
  }

  double classificationError = 1 - double(error) / testData.n_cols;
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(VanillaNetworkTest)
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |     +>|     |     +>|     |
   * +-----+     | +--+--+     | +-----+
   *             |             |
   *  Bias       |  Bias       |
   *  Layer      |  Layer      |
   * +-----+     | +-----+     |
   * |     |     | |     |     |
   * |     +-----+ |     +-----+
   * |     |       |     |
   * +-----+       +-----+
   */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  TestNetwork<>(model, trainData, trainLabels, testData, testLabels, 10, 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model1;
  model1.Add<Linear<> >(dataset.n_rows, 10);
  model1.Add<SigmoidLayer<> >();
  model1.Add<Linear<> >(10, 2);
  model1.Add<LogSoftMax<> >();
  // Vanilla neural net with logistic activation function.
  TestNetwork<>(model1, dataset, labels, dataset, labels, 10, 0.2);
}

BOOST_AUTO_TEST_CASE(ForwardBackwardTest)
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(dataset.n_rows, 50);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(50, 10);
  model.Add<LogSoftMax<> >();

  ens::VanillaUpdate opt;
  model.ResetParameters();
  opt.Initialize(model.Parameters().n_rows, model.Parameters().n_cols);
  double stepSize = 0.01;
  size_t batchSize = 10;

  size_t iteration = 0;
  bool converged = false;
  while (iteration < 100)
  {
    arma::running_stat<double> error;
    size_t batchStart = 0;
    while (batchStart < dataset.n_cols)
    {
      size_t batchEnd = std::min(batchStart + batchSize,
          (size_t) dataset.n_cols);
      arma::mat currentData = dataset.cols(batchStart, batchEnd - 1);
      arma::mat currentLabels = labels.cols(batchStart, batchEnd - 1);
      arma::mat currentResuls;
      model.Forward(currentData, currentResuls);
      arma::mat gradients;
      model.Backward(currentLabels, gradients);
      opt.Update(model.Parameters(), stepSize, gradients);
      batchStart = batchEnd;

      arma::mat prediction = arma::zeros<arma::mat>(1, currentResuls.n_cols);

      for (size_t i = 0; i < currentResuls.n_cols; ++i)
      {
        prediction(i) = arma::as_scalar(arma::find(
            arma::max(currentResuls.col(i)) == currentResuls.col(i), 1)) + 1;
      }

      size_t correct = 0;
      for (size_t i = 0; i < currentLabels.n_cols; i++)
      {
        if (int(arma::as_scalar(prediction.col(i))) ==
            int(arma::as_scalar(currentLabels.col(i))))
        {
          correct++;
        }
      }

      error(1 - (double) correct / batchSize);
    }
    Log::Debug << "Current training error: " << error.mean() << std::endl;
    iteration++;
    if (error.mean() < 0.05)
    {
      converged = true;
      break;
    }
  }

  BOOST_REQUIRE(converged);
}

/**
 * Train the dropout network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DropoutNetworkTest)
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Dropout      Output
   *  Layer         Layer         Layer        Layer
   * +-----+       +-----+       +-----+       +-----+
   * |     |       |     |       |     |       |     |
   * |     +------>|     +------>|     +------>|     |
   * |     |     +>|     |       |     |       |     |
   * +-----+     | +--+--+       +-----+       +-----+
   *             |
   *  Bias       |
   *  Layer      |
   * +-----+     |
   * |     |     |
   * |     +-----+
   * |     |
   * +-----+
   */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  TestNetwork<>(model, trainData, trainLabels, testData, testLabels, 10, 0.1);
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    dataset.col(i) /= norm(dataset.col(i), 2);
  }

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model1;
  model1.Add<Linear<> >(dataset.n_rows, 10);
  model1.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model1.Add<Linear<> >(10, 2);
  model1.Add<LogSoftMax<> >();
  // Vanilla neural net with logistic activation function.
  TestNetwork<>(model1, dataset, labels, dataset, labels, 10, 0.2);
}

/**
 * Train the highway network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(HighwayNetworkTest)
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(dataset.n_rows, 10);
  Highway<>* highway = new Highway<>(10, true);
  highway->Add<Linear<> >(10, 10);
  highway->Add<SigmoidLayer<> >();
  model.Add(highway);
  model.Add<Linear<> >(10, 2);
  model.Add<LogSoftMax<> >();
  TestNetwork<>(model, dataset, labels, dataset, labels, 10, 0.2);
}

/**
 * Train the DropConnect network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DropConnectNetworkTest)
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

 /*
  *  Construct a feed forward network with trainData.n_rows input nodes,
  *  hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
  *  network struct that looks like:
  *
  *  Input         Hidden     DropConnect     Output
  *  Layer         Layer         Layer        Layer
  * +-----+       +-----+       +-----+       +-----+
  * |     |       |     |       |     |       |     |
  * |     +------>|     +------>|     +------>|     |
  * |     |     +>|     |       |     |       |     |
  * +-----+     | +--+--+       +-----+       +-----+
  *             |
  *  Bias       |
  *  Layer      |
  * +-----+     |
  * |     |     |
  * |     +-----+
  * |     |
  * +-----+
  *
  *
  */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<DropConnect<> >(8, 3);
  model.Add<LogSoftMax<> >();

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  TestNetwork<>(model, trainData, trainLabels, testData, testLabels, 10, 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);
  labels += 1;

  FFN<NegativeLogLikelihood<> > model1;
  model1.Add<Linear<> >(dataset.n_rows, 10);
  model1.Add<SigmoidLayer<> >();
  model1.Add<DropConnect<> >(10, 2);
  model1.Add<LogSoftMax<> >();
  // Vanilla neural net with logistic activation function.
  TestNetwork<>(model1, dataset, labels, dataset, labels, 10, 0.2);
}

/**
 * Test miscellaneous things of FFN,
 * e.g. copy/move constructor, assignment operator.
 */
BOOST_AUTO_TEST_CASE(FFNMiscTest)
{
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(2, 3);
  model.Add<ReLULayer<>>();

  auto copiedModel(model);
  copiedModel = model;
  auto movedModel(std::move(model));
  movedModel = std::move(copiedModel);
}

/**
 * Test that serialization works ok.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);

  model.Train(trainData, trainLabels, opt);

  FFN<NegativeLogLikelihood<>> xmlModel, textModel, binaryModel;
  xmlModel.Add<Linear<>>(10, 10); // Layer that will get removed.

  // Serialize into other models.
  SerializeObjectAll(model, xmlModel, textModel, binaryModel);

  arma::mat predictions, xmlPredictions, textPredictions, binaryPredictions;
  model.Predict(testData, predictions);
  xmlModel.Predict(testData, xmlPredictions);
  textModel.Predict(testData, textPredictions);
  textModel.Predict(testData, binaryPredictions);

  CheckMatrices(predictions, xmlPredictions, textPredictions,
      binaryPredictions);
}

/**
 * Test if the custom layers work. The target is to see if the code compiles
 * when the Train and Prediction are called.
 */
BOOST_AUTO_TEST_CASE(CustomLayerTest)
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  FFN<NegativeLogLikelihood<>, RandomInitialization, CustomLayer<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<CustomLayer<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, 15, -1);
  model.Train(trainData, trainLabels, opt);

  arma::mat predictionTemp;
  model.Predict(testData, predictionTemp);
  arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
}

/**
 * Test the overload of Forward function which allows partial forward pass.
 */
BOOST_AUTO_TEST_CASE(PartialForwardTest)
{
  FFN<NegativeLogLikelihood<>, RandomInitialization> model;
  model.Add<Linear<> >(5, 10);

  // Add a new Add<> module which adds a constant term to the input.
  Add<>* addModule = new Add<>(10);
  model.Add(addModule);

  LinearNoBias<>* linearNoBiasModule = new LinearNoBias<>(10, 10);
  model.Add(linearNoBiasModule);

  model.Add<Linear<> >(10, 10);

  model.ResetParameters();
  // Set the parameters of the Add<> module to a matrix of ones.
  addModule->Parameters() = arma::ones(10, 1);
  // Set the parameters of the LinearNoBias<> module to a matrix of ones.
  linearNoBiasModule->Parameters() = arma::ones(10, 10);

  arma::mat input = arma::ones(10, 1);
  arma::mat output;

  // Forward pass only through the Add module.
  model.Forward(input,
                output,
                1 /* Index of the Add module */,
                1 /* Index of the Add module */);

  // As we only forward pass through Add module, input and output should
  // differ by a matrix of ones.
  CheckMatrices(input, output - 1);

  // Forward pass only through the Add module and the LinearNoBias module.
  model.Forward(input,
                output,
                1 /* Index of the Add module */,
                2 /* Index of the LinearNoBias module */);

  // As we only forward pass through Add module followed by the LinearNoBias
  // module, output should be a matrix of 20s.(output = weight * input)
  CheckMatrices(output, arma::ones(10, 1) * 20);
}

/**
 * Test that FFN::Train() returns finite objective value.
 */
BOOST_AUTO_TEST_CASE(FFNTrainReturnObjective)
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  // Vanilla neural net with logistic activation function.
  // Because 92% of the patients are not hyperthyroid the neural
  // network must be significantly better than 92%.
  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 8);
  model.Add<SigmoidLayer<> >();
  model.Add<Dropout<> >();
  model.Add<Linear<> >(8, 3);
  model.Add<LogSoftMax<> >();

  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);

  double objVal = model.Train(trainData, trainLabels, opt);

  BOOST_REQUIRE_EQUAL(std::isfinite(objVal), true);
}
BOOST_AUTO_TEST_SUITE_END();
