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
  const std::vector<double> anchors =
    { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
  const std::vector<std::string> classNames(80);
  YOLOv3Tiny model(imgSize, anchors, classNames);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  model.Predict(testInput, testOutput);

  const size_t expectedRows = (10 * 10 + 20 * 20) * 3 * (5 + 80);
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test number of classes. Other params are set to the default.
 */
TEST_CASE("YOLOv3TinyClasses", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const std::vector<double> anchors =
    { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
  const std::vector<std::string> classNames(80);

  YOLOv3Tiny model(imgSize, anchors, classNames);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  model.Predict(testInput, testOutput);

  const size_t expectedRows = (13 * 13 + 26 * 26) * 3 * (5 + 80);
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test predictions per cell. Other params are set to the default.
 */
TEST_CASE("YOLOv3TinyPredictionsPerCell", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const std::vector<double> anchors =
    { 10, 14, 23, 27 };
  const std::vector<std::string> classNames(80);

  YOLOv3Tiny model(imgSize, anchors, classNames);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  model.Predict(testInput, testOutput);

  const size_t expectedRows = (13 * 13 + 26 * 26) * 3 *
    (5 + 80);
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test incorrect number of anchors.
 */
TEST_CASE("YOLOv3TinyIncorrectAnchors", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const std::vector<double> anchors = { 0, 1, 2, 3, 4, 5, 6, 7 };
  std::vector<std::string> classNames(80);
  REQUIRE_THROWS(YOLOv3Tiny(imgSize, anchors, classNames));
}

/*
 * Test serialize.
 */
TEST_CASE("YOLOv3TinySerialize", "[YOLOv3TinyTest][long]")
{
  const size_t imgSize = 416;
  const size_t predictionsPerCell = 3;
  std::vector<std::string> classNames(80);
  const std::vector<double> anchors =
    { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
  YOLOv3Tiny<arma::mat, EmptyLoss, ConstInitialization>
    model(imgSize, anchors, classNames);

  arma::mat testData(imgSize * imgSize * 3, 1, arma::fill::randu);

  YOLOv3Tiny<arma::mat, EmptyLoss, ConstInitialization>
    xmlModel, jsonModel, binaryModel;
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
  model.Predict(testData, predictions);
  xmlModel.Predict(testData, xmlPredictions);
  jsonModel.Predict(testData, jsonPredictions);
  binaryModel.Predict(testData, binaryPredictions);

  CheckMatrices(predictions, xmlPredictions, jsonPredictions,
      binaryPredictions);
}

/*
 * Test different input image sizes. Other params are set to the default.
 */
TEST_CASE("YOLOv3ImageSize", "[YOLOv3Test][long]")
{
  const size_t imgSize = 320;
  const std::vector<double> anchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
  };

  const std::vector<std::string> classNames(80);
  YOLOv3 model(imgSize, anchors, classNames);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  model.Predict(testInput, testOutput);

  const size_t expectedRows = (10 * 10 + 20 * 20 + 40 * 40) * 3 *
    (5 + classNames.size());
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test number of classes. Other params are set to the default.
 */
TEST_CASE("YOLOv3Classes", "[YOLOv3Test][long]")
{
  const size_t imgSize = 416;
  const std::vector<double> anchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
  };
  const std::vector<std::string> classNames(3);
  YOLOv3 model(imgSize, anchors, classNames);

  arma::mat testInput(imgSize * imgSize * 3, 1);
  arma::mat testOutput;
  model.Predict(testInput, testOutput);

  const size_t expectedRows = (13 * 13 + 26 * 26 + 52 * 52) * 3 *
    (5 + classNames.size());
  REQUIRE(testOutput.n_rows == expectedRows);
}

/*
 * Test incorrect number of anchors.
 */
TEST_CASE("YOLOv3IncorrectAnchors", "[YOLOvTest][long]")
{
  const size_t imgSize = 416;
  const std::vector<double> anchors = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const std::vector<std::string> classNames(80);
  REQUIRE_THROWS(YOLOv3(imgSize, anchors, classNames));
}

/*
 * Test serialize.
 */
TEST_CASE("YOLOv3Serialize", "[YOLOv3Test][long]")
{
  const size_t imgSize = 416;
  const std::vector<double> anchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
  };
  const std::vector<std::string> classNames(80);
  YOLOv3<arma::mat, EmptyLoss, ConstInitialization>
    model(imgSize, anchors, classNames);

  arma::mat testData(imgSize * imgSize * 3, 1, arma::fill::randu);

  YOLOv3<arma::mat, EmptyLoss, ConstInitialization>
    xmlModel, jsonModel, binaryModel;
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  arma::mat predictions, xmlPredictions, jsonPredictions, binaryPredictions;
  model.Predict(testData, predictions);
  xmlModel.Predict(testData, xmlPredictions);
  jsonModel.Predict(testData, jsonPredictions);
  binaryModel.Predict(testData, binaryPredictions);

  CheckMatrices(predictions, xmlPredictions, jsonPredictions,
      binaryPredictions);
}
