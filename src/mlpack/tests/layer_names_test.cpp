/**
 * @file layer_names_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace ann;

BOOST_AUTO_TEST_SUITE(LayerNamesTest);

/**
 * Test if the LayerNameVisitor works properly.
 */
BOOST_AUTO_TEST_CASE(LayerNameVisitorTest)
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
  // Bilinear interpolation is not yet supported by the string converter
  LayerTypes<> unsupportedLayer = new BilinearInterpolation<>();

  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      atrousConvolution) == "atrousconvolution");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      alphaDropout) == "alphadropout");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      batchNorm) == "batchnorm");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      constant) == "constant");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      convolution) == "convolution");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      dropConnect) == "dropconnect");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      dropout) == "dropout");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      flexibleReLU) == "flexiblerelu");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      layerNorm) == "layernorm");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      linear) == "linear");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      linearNoBias) == "linearnobias");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      maxPooling) == "maxpooling");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      meanPooling) == "meanpooling");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      multiplyConstant) == "multiplyconstant");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      reLULayer) == "relu");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      transposedConvolution) == "transposedconvolution");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      identityLayer) == "identity");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      tanHLayer) == "tanh");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      eLU) == "elu");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      hardTanH) == "hardtanh");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      leakyReLU) == "leakyrelu");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      pReLU) == "prelu");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      sigmoidLayer) == "sigmoid");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      logSoftMax) == "logsoftmax");
  BOOST_REQUIRE(boost::apply_visitor(LayerNameVisitor(),
                      unsupportedLayer) == "unsupported");
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
}

BOOST_AUTO_TEST_SUITE_END();
