/**
 * @file tests/ann/layer/batch_norm_serialization.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 *
 * Tests the ann layer modules involving serialization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

// General ANN serialization test.
template<typename LayerType>
void ANNLayerSerializationTest(LayerType& layer)
{
  arma::mat input(5, 100, arma::fill::randu);
  arma::mat output = arma::randi<arma::mat>(1, 100,
      arma::distr_param(0, 4));

  FFN<> model;
  model.Add<Linear>(10);
  model.Add<LayerType>(layer);
  model.Add<ReLU>();
  model.Add<Linear>(5);
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
