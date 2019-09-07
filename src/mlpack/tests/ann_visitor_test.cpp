/**
 * @file ann_visitor_test.cpp
 *
 * Tests for testing visitors in ANN's of mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/visitor/bias_set_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>
#include <mlpack/methods/ann/visitor/reset_visitor.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ANNVisitorTest);

/**
 * Test that the BiasSetVisitor works properly.
 */
BOOST_AUTO_TEST_CASE(BiasSetVisitorTest)
{
  LayerTypes<> linear = new Linear<>(10, 10);

  arma::mat layerWeights(110, 1);
  layerWeights.zeros();

  ResetVisitor resetVisitor;

  boost::apply_visitor(WeightSetVisitor(std::move(layerWeights), 0), linear);

  boost::apply_visitor(resetVisitor, linear);

  arma::mat weight = {"1 2 3 4 5 6 7 8 9 10"};

  size_t biasSize = boost::apply_visitor(BiasSetVisitor(std::move(weight),
      0), linear);

  BOOST_REQUIRE_EQUAL(biasSize, 10);

  arma::mat input(10, 1), output;
  input.randu();

  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(output)),
      linear);

  BOOST_REQUIRE_EQUAL(arma::accu(output), 55);
}

BOOST_AUTO_TEST_SUITE_END();
