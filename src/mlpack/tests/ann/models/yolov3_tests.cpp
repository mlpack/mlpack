/**
 * @file tests/models/yolov3.cpp
 * @author Andrew Furey
 *
 * Tests all the models and layers in models/yolov3/
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING

#include <mlpack.hpp>
#include "../../catch.hpp"
#include "../../serialization.hpp"

using namespace mlpack;

CEREAL_REGISTER_TYPE(mlpack::Layer<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::Identity<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::MultiLayer<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::Convolution<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::BatchNorm<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::LeakyReLU<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::Padding<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::MaxPooling<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::NearestInterpolation<arma::mat>)
CEREAL_REGISTER_TYPE(mlpack::YOLOv3Layer<arma::mat>)

/*
 * Test different input image sizes. Other params are set to the default.
 */
TEST_CASE("YOLOv3TinyImageSize", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 320;
  const size_t numClasses = 80;
  const size_t predictionsPerCell = 3;
  const size_t max = 100;
  const std::vector<double> anchors =
    { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
  YOLOv3Tiny<EmptyLoss, ConstInitialization>
    model(imgSize, numClasses, predictionsPerCell, max, anchors);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  arma::ucolvec numOutputs;
  model.Predict(testInput, testOutput, numOutputs);

  const size_t expectedRows = max * (5 + numClasses);
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test number of classes. Other params are set to the default.
 */
TEST_CASE("YOLOv3TinyClasses", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const size_t numClasses = 3;
  const size_t predictionsPerCell = 3;
  const size_t max = 100;
  const std::vector<double> anchors =
    { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };

  YOLOv3Tiny<EmptyLoss, ConstInitialization>
    model(imgSize, numClasses, predictionsPerCell, max, anchors);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  arma::ucolvec numOutputs;
  model.Predict(testInput, testOutput, numOutputs);

  const size_t expectedRows = max * (5 + numClasses);
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test predictions per cell. Other params are set to the default.
 */
TEST_CASE("YOLOv3TinyPredictionsPerCell", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const size_t numClasses = 80;
  const size_t predictionsPerCell = 1;
  const size_t max = 100;
  const std::vector<double> anchors =
    { 10, 14, 23, 27 };

  YOLOv3Tiny<EmptyLoss, ConstInitialization>
    model(imgSize, numClasses, predictionsPerCell, max, anchors);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  arma::ucolvec numOutputs;
  model.Predict(testInput, testOutput, numOutputs);

  const size_t expectedRows = max * (5 + numClasses);
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test incorrect number of anchors.
 */
TEST_CASE("YOLOv3TinyIncorrectAnchors", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const size_t numClasses = 80;
  const size_t predictionsPerCell = 1;
  const size_t max = 100;
  const std::vector<double> anchors = { 0, 1, 2, 3, 4, 5, 6, 7 };
  REQUIRE_THROWS(YOLOv3Tiny(imgSize, numClasses, predictionsPerCell, max,
                            anchors));
}

/*
 * Test serialize.
 */
TEST_CASE("YOLOv3TinySerialize", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const size_t numClasses = 80;
  const size_t predictionsPerCell = 3;
  const size_t max = 100;
  const std::vector<double> anchors =
    { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
  YOLOv3Tiny<EmptyLoss, ConstInitialization>
    model(imgSize, numClasses, predictionsPerCell, max, anchors);

  arma::mat testData(imgSize * imgSize * 3, 1, arma::fill::randu);

  YOLOv3Tiny<EmptyLoss, ConstInitialization> xmlModel, jsonModel, binaryModel;
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
  arma::ucolvec numPredictions, numXml, numJson, numBinary;
  model.Predict(testData, predictions, numPredictions);
  xmlModel.Predict(testData, xmlPredictions, numXml);
  jsonModel.Predict(testData, jsonPredictions, numJson);
  binaryModel.Predict(testData, binaryPredictions, numBinary);

  CheckMatrices(predictions, xmlPredictions, jsonPredictions,
      binaryPredictions);
}
