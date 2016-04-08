/**
 * @file network_util_test.cpp
 * @author Marcus edel
 *
 * Simple tests for things in the network_util file.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/network_util.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

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

BOOST_AUTO_TEST_SUITE_END();
