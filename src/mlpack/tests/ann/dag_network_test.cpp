/**
 * @file tests/dag_network_test.cpp
 * @author Andrew Furey
 *
 * Tests the DAGNetwork.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
// #ifndef MLPACK_ENABLE_ANN_SERIALIZATION
//   #define MLPACK_ENABLE_ANN_SERIALIZATION
// #endif
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../catch.hpp"

using namespace mlpack;

TEST_CASE("DAGNetworkCheckEmptyGraphTest", "[DAGNetworkTest]")
{
  DAGNetwork<NegativeLogLikelihood, RandomInitialization> model;
  model.InputDimensions() = { 1, 2, 3 };

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::invalid_argument);
}

TEST_CASE("DAGNetworkCheckCycleTest", "[DAGNetworkTest]")
{
  DAGNetwork<NegativeLogLikelihood, RandomInitialization> model;
  model.InputDimensions() = { 1, 2, 3 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);
  model.Connect(layer2, layer3);
  model.Connect(layer3, layer0);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkMultipleOutputsTest", "[DAGNetworkTest]")
{
  DAGNetwork<NegativeLogLikelihood, RandomInitialization> model;
  model.InputDimensions() = { 1, 2, 3 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer1, layer3);
  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkMultipleInputsTest", "[DAGNetworkTest]")
{
  DAGNetwork<NegativeLogLikelihood, RandomInitialization> model;
  model.InputDimensions() = { 1, 2, 3 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);
  size_t layer3 = model.Add<Linear>(10);

  model.Connect(layer0, layer2);
  model.Connect(layer1, layer2);
  model.Connect(layer2, layer3);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkForestTest", "[DAGNetworkTest]")
{
  DAGNetwork<NegativeLogLikelihood, RandomInitialization> model;
  model.InputDimensions() = { 1, 2, 3 };

  size_t layer0 = model.Add<Linear>(10);
  size_t layer1 = model.Add<Linear>(10);
  size_t layer2 = model.Add<Linear>(10);

  size_t layer3 = model.Add<Linear>(10);
  size_t layer4 = model.Add<Linear>(10);
  size_t layer5 = model.Add<Linear>(10);

  model.Connect(layer0, layer1);
  model.Connect(layer1, layer2);

  model.Connect(layer3, layer4);
  model.Connect(layer4, layer5);

  arma::mat testInput = arma::ones(6);
  arma::mat testOutput;

  REQUIRE_THROWS_AS(model.Predict(testInput, testOutput), std::logic_error);
}

TEST_CASE("DAGNetworkDiamondTest", "[DAGNetworkTest]")
{
   /*

       Example that an FFN could not do

                ---- B ----
               /           \
         A ----             --- D
               \           /
                ---- D ----
                \         /
                 ---------
   */

  DAGNetwork<NegativeLogLikelihood, RandomInitialization> dagnet;

  dagnet.InputDimensions() = { 10, 10, 3, 2 };

  size_t layerA = dagnet.Add<Convolution>(1, 3, 3, 1, 1, 1, 1, "none", false);
  size_t layerB = dagnet.Add<Convolution>(1, 1, 1, 1, 1, 0, 0, "none", false);
  size_t layerC = dagnet.Add<Convolution>(1, 1, 1, 1, 1, 0, 0, "none", false);
  size_t layerD = dagnet.Add<Convolution>(2, 3, 3, 1, 1, 1, 1, "none", false);

  dagnet.Connect(layerA, layerD);
  dagnet.Connect(layerA, layerB);
  dagnet.Connect(layerB, layerD);
  dagnet.Connect(layerA, layerC);
  dagnet.Connect(layerC, layerD);

  dagnet.SetAxis(layerD, 2);

  arma::mat testInput = arma::ones(600);
  arma::mat dagnetOutput;

  REQUIRE_NOTHROW(dagnet.Predict(testInput, dagnetOutput));

  std::vector<size_t> expectedOutputDims = { 10, 10, 2, 2 };

  REQUIRE(dagnet.OutputDimensions() == expectedOutputDims);
}
