/**
 * @file tests/layer_names_test.cpp
 * @author Sreenik Seal
 *
 * Tests for testing the string representation of
 * layers in mlpack's ANN module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace ann;

/**
 * Test if the LayerNameVisitor works properly.
 */
TEST_CASE("LayerNameVisitorTest", "[LayerNamesTest]")
{
  LayerTypes<> atrousConvolution = new AtrousConvolution<>();
  LayerTypes<> alphaDropout = new AlphaDropout<>();
  LayerTypes<> batchNorm = new BatchNorm<>();
  LayerTypes<> constant = new Constant<>();
  LayerTypes<> convolution = new Convolution<>();
  LayerTypes<> dropConnect = new DropConnect<>();
  LayerTypes<> dropout = new Dropout<>();
  LayerTypes<> flexibleReLU = new FlexibleReLU<>();
  LayerTypes<> layerNorm = new LayerNorm<>();
  LayerTypes<> linear = new Linear<>();
  LayerTypes<> linearNoBias = new LinearNoBias<>();
  LayerTypes<> maxPooling = new MaxPooling<>();
  LayerTypes<> meanPooling = new MeanPooling<>();
  LayerTypes<> multiplyConstant = new MultiplyConstant<>();
  LayerTypes<> reLULayer = new ReLULayer<>();
  LayerTypes<> transposedConvolution = new TransposedConvolution<>();
  LayerTypes<> identityLayer = new IdentityLayer<>();
  LayerTypes<> tanHLayer = new TanHLayer<>();
  LayerTypes<> eLU = new ELU<>();
  LayerTypes<> hardTanH = new HardTanH<>();
  LayerTypes<> leakyReLU = new LeakyReLU<>();
  LayerTypes<> pReLU = new PReLU<>();
  LayerTypes<> sigmoidLayer = new SigmoidLayer<>();
  LayerTypes<> logSoftMax = new LogSoftMax<>();
  LayerTypes<> lstmLayer = new LSTM<>(100, 10);
  LayerTypes<> creluLayer = new CReLU<>();
  LayerTypes<> highwayLayer = new Highway<>();
  LayerTypes<> gruLayer = new GRU<>();
  LayerTypes<> glimpseLayer = new Glimpse<>();
  LayerTypes<> fastlstmLayer = new FastLSTM<>();
  LayerTypes<> weightnormLayer = new WeightNorm<>(new IdentityLayer<>());

  // Bilinear interpolation is not yet supported by the string converter.
  LayerTypes<> unsupportedLayer = new BilinearInterpolation<>();

  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      atrousConvolution) == "atrousconvolution");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      alphaDropout) == "alphadropout");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      batchNorm) == "batchnorm");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      constant) == "constant");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      convolution) == "convolution");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      dropConnect) == "dropconnect");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      dropout) == "dropout");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      flexibleReLU) == "flexiblerelu");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      layerNorm) == "layernorm");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      linear) == "linear");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      linearNoBias) == "linearnobias");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      maxPooling) == "maxpooling");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      meanPooling) == "meanpooling");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      multiplyConstant) == "multiplyconstant");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      reLULayer) == "relu");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      transposedConvolution) == "transposedconvolution");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      identityLayer) == "identity");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      tanHLayer) == "tanh");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      eLU) == "elu");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      hardTanH) == "hardtanh");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      leakyReLU) == "leakyrelu");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      pReLU) == "prelu");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      sigmoidLayer) == "sigmoid");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      logSoftMax) == "logsoftmax");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      unsupportedLayer) == "unsupported");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      lstmLayer) == "lstm");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      creluLayer) == "crelu");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      highwayLayer) == "highway");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      gruLayer) == "gru");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      glimpseLayer) == "glimpse");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      fastlstmLayer) == "fastlstm");
  REQUIRE(boost::apply_visitor(LayerNameVisitor(),
      weightnormLayer) == "weightnorm");
  // Delete all instances.
  boost::apply_visitor(DeleteVisitor(), atrousConvolution);
  boost::apply_visitor(DeleteVisitor(), alphaDropout);
  boost::apply_visitor(DeleteVisitor(), batchNorm);
  boost::apply_visitor(DeleteVisitor(), constant);
  boost::apply_visitor(DeleteVisitor(), convolution);
  boost::apply_visitor(DeleteVisitor(), dropConnect);
  boost::apply_visitor(DeleteVisitor(), dropout);
  boost::apply_visitor(DeleteVisitor(), flexibleReLU);
  boost::apply_visitor(DeleteVisitor(), layerNorm);
  boost::apply_visitor(DeleteVisitor(), linear);
  boost::apply_visitor(DeleteVisitor(), linearNoBias);
  boost::apply_visitor(DeleteVisitor(), maxPooling);
  boost::apply_visitor(DeleteVisitor(), meanPooling);
  boost::apply_visitor(DeleteVisitor(), multiplyConstant);
  boost::apply_visitor(DeleteVisitor(), reLULayer);
  boost::apply_visitor(DeleteVisitor(), transposedConvolution);
  boost::apply_visitor(DeleteVisitor(), identityLayer);
  boost::apply_visitor(DeleteVisitor(), tanHLayer);
  boost::apply_visitor(DeleteVisitor(), eLU);
  boost::apply_visitor(DeleteVisitor(), hardTanH);
  boost::apply_visitor(DeleteVisitor(), leakyReLU);
  boost::apply_visitor(DeleteVisitor(), pReLU);
  boost::apply_visitor(DeleteVisitor(), sigmoidLayer);
  boost::apply_visitor(DeleteVisitor(), logSoftMax);
  boost::apply_visitor(DeleteVisitor(), unsupportedLayer);
  boost::apply_visitor(DeleteVisitor(), lstmLayer);
  boost::apply_visitor(DeleteVisitor(), creluLayer);
  boost::apply_visitor(DeleteVisitor(), highwayLayer);
  boost::apply_visitor(DeleteVisitor(), gruLayer);
  boost::apply_visitor(DeleteVisitor(), glimpseLayer);
  boost::apply_visitor(DeleteVisitor(), fastlstmLayer);
  boost::apply_visitor(DeleteVisitor(), weightnormLayer);
}
