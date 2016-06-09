/**
 * @file ne_test.cpp
 * @author Bang Liu
 *
 * Test file for NE (Neural Evolution).
 */
#include <cstddef>

#include <mlpack/core.hpp>
#include <mlpack/methods/ne/link_gene.hpp>
#include <mlpack/methods/ne/neuron_gene.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ne;

BOOST_AUTO_TEST_SUITE(NETest);

/**
 * Test LinkGene class.
 */
BOOST_AUTO_TEST_CASE(NELinkGeneTest)
{
  // Create a link gene.
  LinkGene linkGene(1, 2, 0, 10);

  // Test parametric constructor and access functions.
  BOOST_REQUIRE_EQUAL(linkGene.FromNeuronId(), 1);
  BOOST_REQUIRE_EQUAL(linkGene.ToNeuronId(), 2);
  BOOST_REQUIRE_EQUAL(linkGene.InnovationId(), 0);
  BOOST_REQUIRE_EQUAL(linkGene.Weight(), 10);

  // Test set function.
  linkGene.Weight(20);
  BOOST_REQUIRE_EQUAL(linkGene.Weight(), 20);

  // Test copy constructor.
  LinkGene linkGene2(linkGene);
  BOOST_REQUIRE_EQUAL(linkGene2.FromNeuronId(), 1);

  // Test operator =.
  LinkGene linkGene3(1, 3, 1, 7);
  linkGene = linkGene3;
  BOOST_REQUIRE_EQUAL(linkGene3.InnovationId(), 1);

}

/**
 * Test NeuronGene class.
 */
BOOST_AUTO_TEST_CASE(NENeuronGeneTest)
{
  // Create a neuron gene.
  ActivationFuncType actFuncType = SIGMOID;
  NeuronType neuronType = INPUT;
  NeuronGene neuronGene(1, neuronType, actFuncType, 0, 0);

  // Test parametric constructor and access functions.
  BOOST_REQUIRE_EQUAL(neuronGene.Id(), 1);
  BOOST_REQUIRE_EQUAL(neuronGene.Type(), INPUT);
  BOOST_REQUIRE_EQUAL(neuronGene.ActFuncType(), SIGMOID);
  BOOST_REQUIRE_EQUAL(neuronGene.aInput, 0);
  BOOST_REQUIRE_EQUAL(neuronGene.aActivation, 0);

  // Test set function
  neuronGene.ActFuncType(RELU);
  BOOST_REQUIRE_EQUAL(neuronGene.ActFuncType(), RELU);

  // Test copy function.
  NeuronGene neuronGene2(neuronGene);
  BOOST_REQUIRE_EQUAL(neuronGene2.Id(), 1);

  // Test operator =.
  NeuronGene neuronGene3(11, neuronType, actFuncType, 0, 0);
  neuronGene = neuronGene3;
  BOOST_REQUIRE_EQUAL(neuronGene.Id(), 11);

}

BOOST_AUTO_TEST_SUITE_END();
