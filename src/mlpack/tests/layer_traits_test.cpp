/**
 * @file layer_traits_test.cpp
 * @author Marcus Edel
 *
 * Test the LayerTraits class. Because all of the values are known at compile
 * time, this test is meant to ensure that uses of LayerTraits still compile
 * okay and react as expected.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/multiclass_classification_layer.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(LayerTraitsTest);

// Test the defaults.
BOOST_AUTO_TEST_CASE(DefaultsTraitsTest)
{
  // An irrelevant non-connection type class is used here so that the default
  // implementation of ConnectionTraits is chosen.
  bool b = LayerTraits<int>::IsBinary;
  BOOST_REQUIRE_EQUAL(b, false);

  b =  LayerTraits<int>::IsOutputLayer;
  BOOST_REQUIRE_EQUAL(b, false);

  b =  LayerTraits<int>::IsBiasLayer;
  BOOST_REQUIRE_EQUAL(b, false);
}

// Test the BiasLayer traits.
BOOST_AUTO_TEST_CASE(BiasLayerTraitsTest)
{
  bool b = LayerTraits<BiasLayer<> >::IsBinary;
  BOOST_REQUIRE_EQUAL(b, false);

  b = LayerTraits<BiasLayer<> >::IsOutputLayer;
  BOOST_REQUIRE_EQUAL(b, false);

  b = LayerTraits<BiasLayer<> >::IsBiasLayer;
  BOOST_REQUIRE_EQUAL(b, true);
}

// Test the MulticlassClassificationLayer traits.
BOOST_AUTO_TEST_CASE(MulticlassClassificationLayerTraitsTest)
{
  bool b = LayerTraits<MulticlassClassificationLayer<> >::IsBinary;
  BOOST_REQUIRE_EQUAL(b, false);

  b = LayerTraits<MulticlassClassificationLayer<> >::IsOutputLayer;
  BOOST_REQUIRE_EQUAL(b, true);

  b = LayerTraits<MulticlassClassificationLayer<> >::IsBiasLayer;
  BOOST_REQUIRE_EQUAL(b, false);
}

BOOST_AUTO_TEST_SUITE_END();
