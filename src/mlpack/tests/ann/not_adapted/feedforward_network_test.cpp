/**
 * @file tests/feedforward_network_test.cpp
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

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <ensmallen.hpp>

#include "../catch.hpp"
#include "../serialization.hpp"

using namespace mlpack;

/**
 * Check whether copying and moving network with Reparametrization is working or not.
 *
TEST_CASE("CheckCopyMovingReparametrizationNetworkTest",
          "[FeedForwardNetworkTest]")
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("thyroid_train.csv", trainData, true);

  arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
  trainData.shed_row(trainData.n_rows - 1);

  // Construct a feed forward network with trainData.n_rows input nodes,
  // followed by a linear layer and then a reparametrization layer.
  FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
  model->Add<Linear>(8);
  model->Add<Reparametrization>(false, true, 1);
  model->Add<LogSoftMax>();

  FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
  model1->Add<Linear>(8);
  model1->Add<Reparametrization>(false, true, 1);
  model1->Add<LogSoftMax>();

  // Check whether copy constructor is working or not.
  CheckCopyFunction(model, trainData, trainLabels);

  // Check whether move constructor is working or not.
  CheckMoveFunction(model1, trainData, trainLabels, 1);
}
*/

/**
 * Check whether copying and moving network with Reparametrization is working or not.
 */
// TEST_CASE("CheckCopyMovingReparametrizationNetworkTest",
//           "[FeedForwardNetworkTest]")
// {
//   // Load the dataset.
//   arma::mat trainData;
//   data::Load("thyroid_train.csv", trainData, true);
//
//   // Normalize labels to [0, 2].
//   arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
//   trainData.shed_row(trainData.n_rows - 1);
//
//   /*
//    * Construct a feed forward network with trainData.n_rows input nodes,
//    * followed by a linear layer and then a reparametrization layer.
//    */
//
//   FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
//   model->Add<Linear<> >(trainData.n_rows, 8);
//   model->Add<Reparametrization<> >(4, false, true, 1);
//   model->Add<LogSoftMax<> >();
//
//   FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
//   model1->Add<Linear<> >(trainData.n_rows, 8);
//   model1->Add<Reparametrization<> >(4, false, true, 1);
//   model1->Add<LogSoftMax<> >();
//
//   // Check whether copy constructor is working or not.
//   CheckCopyFunction<>(model, trainData, trainLabels, 1);
//
//   // Check whether move constructor is working or not.
//   CheckMoveFunction<>(model1, trainData, trainLabels, 1);
// }

/**
 * Check whether copying and moving network with Reparametrization is working or not.
 */
// TEST_CASE("CheckCopyMovingReparametrizationNetworkTestNoBias",
//           "[FeedForwardNetworkTest]")
// {
//   // Load the dataset.
//   arma::mat trainData;
//   data::Load("thyroid_train.csv", trainData, true);
//
//   // Normalize labels to [0, 2].
//   arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
//   trainData.shed_row(trainData.n_rows - 1);
//
//   /*
//    * Construct a feed forward network with trainData.n_rows input nodes,
//    * followed by a linear layer and then a reparametrization layer.
//    */
//
//   FFN<NegativeLogLikelihood> *model = new FFN<NegativeLogLikelihood>;
//   model->Add<LinearNoBias<> >(trainData.n_rows, 8);
//   model->Add<Reparametrization<> >(4, false, true, 1);
//   model->Add<LogSoftMax<> >();
//
//   FFN<NegativeLogLikelihood> *model1 = new FFN<NegativeLogLikelihood>;
//   model1->Add<LinearNoBias<> >(trainData.n_rows, 8);
//   model1->Add<Reparametrization<> >(4, false, true, 1);
//   model1->Add<LogSoftMax<> >();
//
//   // Check whether copy constructor is working or not.
//   CheckCopyFunction<>(model, trainData, trainLabels, 1);
//
//   // Check whether move constructor is working or not.
//   CheckMoveFunction<>(model1, trainData, trainLabels, 1);
// }

/**
 * Train the highway network on a larger dataset.
 *
TEST_CASE("HighwayNetworkTest", "[FeedForwardNetworkTest]")
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::mat labels = arma::zeros(1, dataset.n_cols);
  labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1);

  FFN<NegativeLogLikelihood> model;
  model.Add<Linear>(10);
  Highway* highway = new Highway();
  highway->Add<Linear>(10);
  highway->Add<Sigmoid>();
  model.Add(highway); // This takes ownership of the memory.
  model.Add<Linear>(2);
  model.Add<LogSoftMax>();
  TestNetwork(model, dataset, labels, dataset, labels, 10, 0.2);
}*/

/**
 * Test that serialization works ok for PReLU.
 */
// TEST_CASE("PReLUSerializationTest", "[FeedForwardNetworkTest]")
// {
//   // Load the dataset.
//   arma::mat trainData;
//   if (!data::Load("thyroid_train.csv", trainData))
//     FAIL("Cannot open thyroid_train.csv");
//
//   arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
//   trainData.shed_row(trainData.n_rows - 1);
//   trainLabels -= 1; // The labels should be between 0 and numClasses - 1.
//
//   arma::mat testData;
//   if (!data::Load("thyroid_test.csv", testData))
//     FAIL("Cannot load dataset thyroid_test.csv");
//
//   arma::mat testLabels = testData.row(testData.n_rows - 1);
//   testData.shed_row(testData.n_rows - 1);
//   testLabels -= 1; // The labels should be between 0 and numClasses - 1.
//
//   // Vanilla neural net with logistic activation function.
//   // Because 92% of the patients are not hyperthyroid the neural
//   // network must be significant better than 92%.
//   FFN<NegativeLogLikelihood> model;
//   model.Add<Linear<> >(trainData.n_rows, 8);
//   model.Add<PReLU<> >();
//   model.Add<Dropout<> >();
//   model.Add<Linear<> >(8, 3);
//   model.Add<LogSoftMax<> >();
//
//   ens::RMSProp opt(0.01, 32, 0.88, 1e-8, trainData.n_cols /* 1 epoch */, -1);
//
//   model.Train(trainData, trainLabels, opt);
//
//   FFN<NegativeLogLikelihood> xmlModel, jsonModel, binaryModel;
//   xmlModel.Add<Linear<>>(10, 10); // Layer that will get removed.
//
//   // Serialize into other models.
//   SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);
//
//   arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
//   model.Predict(testData, predictions);
//   xmlModel.Predict(testData, xmlPredictions);
//   jsonModel.Predict(testData, jsonPredictions);
//   jsonModel.Predict(testData, binaryPredictions);
//
//   CheckMatrices(predictions, xmlPredictions, jsonPredictions,
//       binaryPredictions);
// }

/**
 * Test if the custom layers work. The target is to see if the code compiles
 * when the Train and Prediction are called.
 */
// TEST_CASE("CustomLayerTest", "[FeedForwardNetworkTest]")
// {
//   // Load the dataset.
//   arma::mat trainData;
//   if (!data::Load("thyroid_train.csv", trainData))
//     FAIL("Cannot open thyroid_train.csv");
//
//   arma::mat trainLabels = trainData.row(trainData.n_rows - 1);
//   trainData.shed_row(trainData.n_rows - 1);
//   trainLabels -= 1; // The labels should be between 0 and numClasses - 1.
//
//   arma::mat testData;
//   if (!data::Load("thyroid_test.csv", testData))
//     FAIL("Cannot load dataset thyroid_test.csv");
//
//   arma::mat testLabels = testData.row(testData.n_rows - 1);
//   testData.shed_row(testData.n_rows - 1);
//   testLabels -= 1; // The labels should be between 0 and numClasses - 1.
//
//   FFN<NegativeLogLikelihood, RandomInitialization, CustomLayer<> > model;
//   model.Add<Linear<> >(trainData.n_rows, 8);
//   model.Add<CustomLayer<> >();
//   model.Add<Linear<> >(8, 3);
//   model.Add<LogSoftMax<> >();
//
//   ens::RMSProp opt(0.01, 32, 0.88, 1e-8, 15, -1);
//   model.Train(trainData, trainLabels, opt);
//
//   arma::mat predictionTemp;
//   model.Predict(testData, predictionTemp);
//   arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);
// }
