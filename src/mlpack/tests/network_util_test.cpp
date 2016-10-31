/**
 * @file network_util_test.cpp
 * @author Marcus Edel
 *
 * Simple tests for things in the network_util file.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/network_util.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(NetworkUtilTest);

/**
 * Test the network size auxiliary function.
 */
BOOST_AUTO_TEST_CASE(NetworkSizeTest)
{
  // Create a two layer network without weights.
  BaseLayer<> baseLayer1;
  BaseLayer<> baseLayer2;
  auto noneWeightNetwork = std::tie(baseLayer1, baseLayer2);

  BOOST_REQUIRE_EQUAL(NetworkSize(noneWeightNetwork), 0);

  // Create a two layer network.
  LinearLayer<> linearLayer1(10, 10);
  LinearLayer<> linearLayer2(10, 100);

  // Reuse the layer form the first network.
  auto weightNetwork = std::tie(linearLayer1, baseLayer1, linearLayer2,
      baseLayer2);

  BOOST_REQUIRE_EQUAL(NetworkSize(weightNetwork), 1100);
}

/**
 * Test the layer size auxiliary function.
 */
BOOST_AUTO_TEST_CASE(LayerSizeTest)
{
  // Create layer without weights.
  BaseLayer<> baseLayer;
  BOOST_REQUIRE_EQUAL(LayerSize(baseLayer, baseLayer.OutputParameter()), 0);

  // Create layer with weights.
  LinearLayer<> linearLayer(10, 10);
  BOOST_REQUIRE_EQUAL(LayerSize(linearLayer,
      linearLayer.OutputParameter()), 100);
}

/**
 * Test the network input size auxiliary function.
 */
BOOST_AUTO_TEST_CASE(NetworkInputSizeTest)
{
  // Create a two layer network without weights.
  BaseLayer<> baseLayer1;
  BaseLayer<> baseLayer2;
  auto noneWeightNetwork = std::tie(baseLayer1, baseLayer2);

  BOOST_REQUIRE_EQUAL(NetworkInputSize(noneWeightNetwork), 0);

  // Create a two layer network.
  LinearLayer<> linearLayer1(5, 10);
  LinearLayer<> linearLayer2(10, 100);

  // Reuse the layer form the first network.
  auto weightNetwork = std::tie(linearLayer1, baseLayer1, linearLayer2,
      baseLayer2);

  BOOST_REQUIRE_EQUAL(NetworkInputSize(weightNetwork), 5);
}

/**
 * Test the layer input size auxiliary function.
 */
BOOST_AUTO_TEST_CASE(LayerInputSizeTest)
{
  // Create layer without weights.
  BaseLayer<> baseLayer;
  BOOST_REQUIRE_EQUAL(LayerInputSize(baseLayer,
    baseLayer.OutputParameter()), 0);

  // Create layer with weights.
  LinearLayer<> linearLayer(5, 10);
  BOOST_REQUIRE_EQUAL(LayerInputSize(linearLayer,
      linearLayer.OutputParameter()), 5);
}

/**
 * Test the network weight auxiliary function using the given initialization
 * rule.
 */
BOOST_AUTO_TEST_CASE(NetworkWeightsInitTest)
{
  // Create a two layer network.
  LinearLayer<> linearLayer1(10, 10);
  LinearLayer<> linearLayer2(10, 100);

  arma::mat parameter = arma::zeros<arma::mat>(1100, 1);

  // Create the network.
  auto network = std::tie(linearLayer1, linearLayer2);

  BOOST_REQUIRE_EQUAL(arma::accu(parameter), 0);

  RandomInitialization constantInit(1, 1);
  NetworkWeights(constantInit, parameter, network);

  BOOST_REQUIRE_EQUAL(arma::accu(linearLayer1.Weights()), 100);
  BOOST_REQUIRE_EQUAL(arma::accu(linearLayer2.Weights()), 1000);
  BOOST_REQUIRE_EQUAL(arma::accu(parameter), 1100);
}

/**
 * Test the layer weight auxiliary function using the given initialization rule.
 */
BOOST_AUTO_TEST_CASE(LayerWeightsInitTest)
{
  // Create a two layer network.
  LinearLayer<> linearLayer1(10, 10);

  arma::mat parameter = arma::zeros<arma::mat>(100, 1);

  BOOST_REQUIRE_EQUAL(arma::accu(parameter), 0);

  RandomInitialization constantInit(1, 1);
  arma::mat output;
  LayerWeights(constantInit, linearLayer1, parameter, 0, output);

  BOOST_REQUIRE_EQUAL(arma::accu(linearLayer1.Weights()), 100);
  BOOST_REQUIRE_EQUAL(arma::accu(parameter), 100);
}

BOOST_AUTO_TEST_SUITE_END();
