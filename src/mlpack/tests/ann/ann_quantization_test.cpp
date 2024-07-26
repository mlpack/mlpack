/**
 * @file ann_quantization_test.cpp
 * @author Mark Fischinger 
 *
 * Tests for the quantization functionality in FFN and RNN classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/lstm.hpp>
#include <mlpack/methods/ann/layer/recurrent.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/core/data/binarize.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNQuantizationTest);

/**
 * Test that quantization of FFN preserves network structure.
 */
BOOST_AUTO_TEST_CASE(FFNQuantizationStructureTest)
{
  // Create a simple FFN.
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(10, 5);
  model.Add<Linear<>>(5, 2);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Check that the quantized model has the same structure.
  BOOST_REQUIRE_EQUAL(model.Model().size(), quantizedModel.Model().size());
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[0]->OutputSize(), 5);
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[1]->OutputSize(), 2);
}

/**
 * Test that quantization of FFN preserves weight dimensions.
 */
BOOST_AUTO_TEST_CASE(FFNQuantizationWeightDimensionsTest)
{
  // Create a simple FFN with known weight dimensions.
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(10, 5);
  model.Add<Linear<>>(5, 2);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Check that weight dimensions are preserved.
  BOOST_REQUIRE_EQUAL(model.Model()[0]->Parameters().n_rows,
                      quantizedModel.Model()[0]->Parameters().n_rows);
  BOOST_REQUIRE_EQUAL(model.Model()[0]->Parameters().n_cols,
                      quantizedModel.Model()[0]->Parameters().n_cols);
  BOOST_REQUIRE_EQUAL(model.Model()[1]->Parameters().n_rows,
                      quantizedModel.Model()[1]->Parameters().n_cols);
  BOOST_REQUIRE_EQUAL(model.Model()[1]->Parameters().n_cols,
                      quantizedModel.Model()[1]->Parameters().n_cols);
}

/**
 * Test that quantization of FFN results in int8_t weights.
 */
BOOST_AUTO_TEST_CASE(FFNQuantizationWeightTypeTest)
{
  // Create a simple FFN.
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(10, 5);
  model.Add<Linear<>>(5, 2);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Check that weights are of type int8_t.
  BOOST_REQUIRE_EQUAL(typeid(quantizedModel.Model()[0]->Parameters().at(0, 0)),
                      typeid(int8_t));
  BOOST_REQUIRE_EQUAL(typeid(quantizedModel.Model()[1]->Parameters().at(0, 0)),
                      typeid(int8_t));
}

/**
 * Test that quantization of RNN preserves network structure.
 */
BOOST_AUTO_TEST_CASE(RNNQuantizationStructureTest)
{
  // Create a simple RNN.
  RNN<MeanSquaredError<>> model(5, 3, 10);
  model.Add<LSTM<>>(3, 4);
  model.Add<Linear<>>(4, 2);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Check that the quantized model has the same structure.
  BOOST_REQUIRE_EQUAL(model.Model().size(), quantizedModel.Model().size());
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[0]->OutputSize(), 4);
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[1]->OutputSize(), 2);
}

/**
 * Test that quantization of RNN preserves weight dimensions.
 */
BOOST_AUTO_TEST_CASE(RNNQuantizationWeightDimensionsTest)
{
  // Create a simple RNN with known weight dimensions.
  RNN<MeanSquaredError<>> model(5, 3, 10);
  model.Add<LSTM<>>(3, 4);
  model.Add<Linear<>>(4, 2);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Check that weight dimensions are preserved.
  BOOST_REQUIRE_EQUAL(model.Model()[0]->Parameters().n_rows,
                      quantizedModel.Model()[0]->Parameters().n_rows);
  BOOST_REQUIRE_EQUAL(model.Model()[0]->Parameters().n_cols,
                      quantizedModel.Model()[0]->Parameters().n_cols);
  BOOST_REQUIRE_EQUAL(model.Model()[1]->Parameters().n_rows,
                      quantizedModel.Model()[1]->Parameters().n_cols);
  BOOST_REQUIRE_EQUAL(model.Model()[1]->Parameters().n_cols,
                      quantizedModel.Model()[1]->Parameters().n_cols);
}

/**
 * Test that quantization of RNN results in int8_t weights.
 */
BOOST_AUTO_TEST_CASE(RNNQuantizationWeightTypeTest)
{
  // Create a simple RNN.
  RNN<MeanSquaredError<>> model(5, 3, 10);
  model.Add<LSTM<>>(3, 4);
  model.Add<Linear<>>(4, 2);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Check that weights are of type int8_t.
  BOOST_REQUIRE_EQUAL(typeid(quantizedModel.Model()[0]->Parameters().at(0, 0)),
                      typeid(int8_t));
  BOOST_REQUIRE_EQUAL(typeid(quantizedModel.Model()[1]->Parameters().at(0, 0)),
                      typeid(int8_t));
}

/**
 * Test that quantization preserves the general behavior of the network.
 */
BOOST_AUTO_TEST_CASE(QuantizationBehaviorTest)
{
  // Create a simple FFN.
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(5, 10);
  model.Add<ReLU<>>();
  model.Add<Linear<>>(10, 3);
  model.Add<LogSoftMax<>>();

  // Generate some random input data.
  arma::mat input = arma::randu<arma::mat>(5, 100);
  arma::mat output;

  // Get output from the original model.
  model.Predict(input, output);

  // Quantize the network.
  auto quantizedModel = model.Quantize<arma::Mat<int8_t>>();

  // Get output from the quantized model.
  arma::mat quantizedOutput;
  quantizedModel.Predict(input, quantizedOutput);

  // Check that the outputs are similar (allowing for some quantization error).
  BOOST_REQUIRE_LE(arma::norm(output - quantizedOutput), 0.1);
}

/**
 * Test the standalone Quantize function with FFN.
 */
BOOST_AUTO_TEST_CASE(StandaloneQuantizeFunctionFFNTest)
{
  // Create a simple FFN.
  FFN<MeanSquaredError<>> model;
  model.Add<Linear<>>(10, 5);
  model.Add<Linear<>>(5, 2);

  // Use the standalone Quantize function.
  auto quantizedModel = Quantize<arma::Mat<int8_t>>(model);

  // Check that the quantized model has the same structure.
  BOOST_REQUIRE_EQUAL(model.Model().size(), quantizedModel.Model().size());
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[0]->OutputSize(), 5);
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[1]->OutputSize(), 2);

  // Check that weights are of type int8_t.
  BOOST_REQUIRE_EQUAL(typeid(quantizedModel.Model()[0]->Parameters().at(0, 0)),
                      typeid(int8_t));
}

/**
 * Test the standalone Quantize function with RNN.
 */
BOOST_AUTO_TEST_CASE(StandaloneQuantizeFunctionRNNTest)
{
  // Create a simple RNN.
  RNN<MeanSquaredError<>> model(5, 3, 10);
  model.Add<LSTM<>>(3, 4);
  model.Add<Linear<>>(4, 2);

  // Use the standalone Quantize function.
  auto quantizedModel = Quantize<arma::Mat<int8_t>>(model);

  // Check that the quantized model has the same structure.
  BOOST_REQUIRE_EQUAL(model.Model().size(), quantizedModel.Model().size());
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[0]->OutputSize(), 4);
  BOOST_REQUIRE_EQUAL(quantizedModel.Model()[1]->OutputSize(), 2);

  // Check that weights are of type int8_t.
  BOOST_REQUIRE_EQUAL(typeid(quantizedModel.Model()[0]->Parameters().at(0, 0)),
                      typeid(int8_t));
}

BOOST_AUTO_TEST_SUITE_END();