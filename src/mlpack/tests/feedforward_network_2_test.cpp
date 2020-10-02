/**
 * @file tests/feedforward_network_2_test.cpp
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
#include <mlpack/methods/kmeans/kmeans.hpp>

#include <ensmallen.hpp>

#include "catch.hpp"
#include "serialization_catch.hpp"
#include "custom_layer.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::kmeans;


/**
 * Train the highway network on a larger dataset.
 */
TEST_CASE("HighwayNetworkTest", "[FeedForwardNetworkTest]")
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
  model.Add(highway); // This takes ownership of the memory.
  model.Add<Linear<> >(10, 2);
  model.Add<LogSoftMax<> >();
  TestNetwork<>(model, dataset, labels, dataset, labels, 10, 0.2);
}

/**
 * Train the DropConnect network on a larger dataset.
 */
TEST_CASE("DropConnectNetworkTest", "[FeedForwardNetworkTest]")
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
TEST_CASE("FFNMiscTest", "[FeedForwardNetworkTest]")
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
TEST_CASE("FFSerializationTest", "[FeedForwardNetworkTest]")
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

  FFN<NegativeLogLikelihood<>> xmlModel, jsonModel, binaryModel;
  xmlModel.Add<Linear<>>(10, 10); // Layer that will get removed.

  // Serialize into other models.
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
  model.Predict(testData, predictions);
  xmlModel.Predict(testData, xmlPredictions);
  jsonModel.Predict(testData, jsonPredictions);
  jsonModel.Predict(testData, binaryPredictions);

  CheckMatrices(predictions, xmlPredictions, jsonPredictions,
      binaryPredictions);
}

/**
 * Test if the custom layers work. The target is to see if the code compiles
 * when the Train and Prediction are called.
 */
TEST_CASE("CustomLayerTest", "[FeedForwardNetworkTest]")
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
TEST_CASE("PartialForwardTest", "[FeedForwardNetworkTest]")
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
TEST_CASE("FFNTrainReturnObjective", "[FeedForwardNetworkTest]")
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

  REQUIRE(std::isfinite(objVal) == true);
}

/**
 * Test that FFN::Model() allows us to access the instantiated network.
 */
TEST_CASE("FFNReturnModel", "[FeedForwardNetworkTest]")
{
  // Create dummy network.
  FFN<NegativeLogLikelihood<> > model;
  Linear<>* linearA = new Linear<>(3, 3);
  model.Add(linearA);
  Linear<>* linearB = new Linear<>(3, 4);
  model.Add(linearB);

  // Initialize network parameter.
  model.ResetParameters();

  // Set all network parameter to one.
  model.Parameters().ones();

  // Zero the second layer parameter.
  linearB->Parameters().zeros();

  // Get the layer parameter from layer A and layer B and store them in
  // parameterA and parameterB.
  arma::mat parameterA, parameterB;
  boost::apply_visitor(ParametersVisitor(parameterA), model.Model()[0]);
  boost::apply_visitor(ParametersVisitor(parameterB), model.Model()[1]);

  CheckMatrices(parameterA, arma::ones(3 * 3 + 3, 1));
  CheckMatrices(parameterB, arma::zeros(3 * 4 + 4, 1));

  CheckMatrices(linearA->Parameters(), arma::ones(3 * 3 + 3, 1));
  CheckMatrices(linearB->Parameters(), arma::zeros(3 * 4 + 4, 1));
}

/**
 * Test to see if the FFN code compiles when the Optimizer
 * doesn't have the MaxIterations() method.
 */
TEST_CASE("OptimizerTest", "[FeedForwardNetworkTest]")
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

  ens::DE opt(200, 1000, 0.6, 0.8, 1e-5);
  model.Train(trainData, trainLabels, opt);
}

/**
 * Train the RBF network on a larger dataset.
 */
TEST_CASE("RBFNetworkTest", "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  arma::mat trainLabels1 = arma::zeros(3, trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    trainLabels1.col(i).row((trainLabels(i) - 1)) = 1;
  }

  arma::mat testData;
  data::Load("thyroid_test.csv", testData, true);

  arma::mat testLabels = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         RBF          Activation    Output
   *  Layer         Layer         Layer        Layer
   * +-----+       +-----+       +-----+       +-----+
   * |     |       |     |       |     |       |     |
   * |     +------>|     +------>|     +------>|     |
   * |     |       |     |       |     |       |     |
   * +-----+       +--+--+       +-----+       +-----+
   */
  arma::mat centroids;
  KMeans<> kmeans;
  kmeans.Cluster(trainData, 8, centroids);

  FFN<MeanSquaredError<> > model;
  model.Add<RBF<> >(trainData.n_rows, 8, centroids);
  model.Add<Linear<> >(8, 3);

  // RBFN neural net with MeanSquaredError.
  TestNetwork<>(model, trainData, trainLabels1, testData, testLabels, 10, 0.1);

  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    dataset.col(i) /= norm(dataset.col(i), 2);
  }

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  arma::mat labels1 = arma::zeros(2, dataset.n_cols);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    labels1.col(i).row(labels(i)) = 1;
  }
  labels += 1;

  arma::mat centroids1;
  arma::Row<size_t> assignments;
  KMeans<> kmeans1;
  kmeans1.Cluster(dataset, 140, centroids1);

  FFN<MeanSquaredError<> > model1;
  model1.Add<RBF<> >(dataset.n_rows, 140, centroids1, 4.1);
  model1.Add<Linear<> >(140, 2);

  // RBFN neural net with MeanSquaredError.
  TestNetwork<>(model1, dataset, labels1, dataset, labels, 10, 0.1);
}
