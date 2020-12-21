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

#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/flexible_relu.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <ensmallen.hpp>

#include "catch.hpp"
//#include "serialization.hpp"
//#include "custom_layer.hpp"

using namespace mlpack;
using namespace mlpack::ann;

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
  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, 10001, -100);
  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t correct = arma::accu(prediction == testLabels);

  double classificationError = 1 - double(correct) / testData.n_cols;
  /* REQUIRE(classificationError <= classificationErrorThreshold); */
}

/**
 * Train the vanilla network on a larger dataset.
 */
TEST_CASE("FFVanillaNetworkTest", "[FeedForwardNetworkTest]")
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

  double elapsedSecondsAverage = 0;
  const size_t trails = 10;
  for (size_t i = 0; i < trails; ++i)
  {
    auto start = std::chrono::steady_clock::now();
    FFN<> model;
    model.Add<Linear<> >(trainData.n_rows, 128);
    model.Add<FlexibleReLU<> >();
    model.Add<Linear<> >(128, 256);
    model.Add<Linear<> >(256, 256);
    model.Add<Linear<> >(256, 256);
    model.Add<Linear<> >(256, 256);
    model.Add<Linear<> >(256, 512);
    model.Add<Linear<> >(512, 2048);
    model.Add<Linear<> >(2048, 512);
    model.Add<Linear<> >(512, 8);
    model.Add<Linear<> >(8, 3);
    model.Add<LogSoftMax<> >();

    // Vanilla neural net with logistic activation function.
    // Because 92% of the patients are not hyperthyroid the neural
    // network must be significant better than 92%.
    TestNetwork<>(model, trainData, trainLabels, testData, testLabels, 10, 0.1);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end-start;
    std::cout << "elapsed time: " << elapsedSeconds.count() << "s\n";

    elapsedSecondsAverage += elapsedSeconds.count();
  }

  std::cout << "--------------------------------------\n";
  std::cout << "elapsed time averaged(" << trails << "): "
      << elapsedSecondsAverage / (double) trails << "s\n";


  /* arma::mat dataset; */
  /* dataset.load("mnist_first250_training_4s_and_9s.arm"); */

  /* // Normalize each point since these are images. */
  /* for (size_t i = 0; i < dataset.n_cols; ++i) */
  /*   dataset.col(i) /= norm(dataset.col(i), 2); */

  /* arma::mat labels = arma::zeros(1, dataset.n_cols); */
  /* labels.submat(0, labels.n_cols / 2, 0, labels.n_cols - 1).fill(1); */
  /* labels += 1; */

  /* FFN<NegativeLogLikelihood<> > model1; */
  /* model1.Add<Linear<> >(dataset.n_rows, 10); */
  /* model1.Add<SigmoidLayer<> >(); */
  /* model1.Add<Linear<> >(10, 2); */
  /* model1.Add<LogSoftMax<> >(); */
  /* // Vanilla neural net with logistic activation function. */
  /* TestNetwork<>(model1, dataset, labels, dataset, labels, 10, 0.2); */
}

