/**
 * @file ne_test.cpp
 * @author Bang Liu
 *
 * Test file for NE (Neural Evolution).
 */
#include <cstddef>

#include <mlpack/core.hpp>
#include <mlpack/methods/ne/utils.hpp>
#include <mlpack/methods/ne/parameters.hpp>
#include <mlpack/methods/ne/tasks.hpp>
#include <mlpack/methods/ne/link_gene.hpp>
#include <mlpack/methods/ne/neuron_gene.hpp>
#include <mlpack/methods/ne/genome.hpp>
#include <mlpack/methods/ne/species.hpp>
#include <mlpack/methods/ne/cne.hpp>

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
  NeuronGene neuronGene(1, INPUT, SIGMOID, 0, 0);

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
  NeuronGene neuronGene3(11, INPUT, SIGMOID, 0, 0);
  neuronGene = neuronGene3;
  BOOST_REQUIRE_EQUAL(neuronGene.Id(), 11);

}

/**
 * Test Genome class.
 */
BOOST_AUTO_TEST_CASE(NEGenomeTest)
{

}

/**
 * Test Species class.
 */
BOOST_AUTO_TEST_CASE(NESpeciesTest)
{

}

/**
 * Test CNE by XOR task.
 */
BOOST_AUTO_TEST_CASE(NECneXorTest)
{
  mlpack::math::RandomSeed(1);

  // Set CNE algorithm parameters.
  Parameters params;
  params.aSpeciesSize = 500;
  params.aMutateRate = 0.1;
  params.aMutateSize = 0.02;
  params.aElitePercentage = 0.2;
  //params.aCrossoverRate = 0.3;
  params.aMaxGeneration = 1000;

  // Construct seed genome for xor task.
  size_t id = 0;
  size_t numInput = 3;
  size_t numOutput = 1;
  size_t depth = 2;
  double fitness = -1;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, SIGMOID, 0, 0);
  NeuronGene inputGene2(1, INPUT, SIGMOID, 0, 0);
  NeuronGene biasGene(2, BIAS, LINEAR, 0, 0);
  NeuronGene outputGene(3, OUTPUT, SIGMOID, 0, 0);
  NeuronGene hiddenGene(4, HIDDEN, SIGMOID, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 3, 0, 0);
  LinkGene link2(1, 3, 0, 0);
  LinkGene link3(2, 3, 0, 0);
  LinkGene link4(0, 4, 0, 0);
  LinkGene link5(1, 4, 0, 0);
  LinkGene link6(2, 4, 0, 0);
  LinkGene link7(4, 3, 0, 0);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);
  linkGenes.push_back(link7);

  Genome seedGenome = Genome(0, neuronGenes, linkGenes, numInput, numOutput, depth, fitness);

  // Specify task type.
  TaskXor<ann::MeanSquaredErrorFunction> task;

  // Construct CNE instance.
  CNE<TaskXor<ann::MeanSquaredErrorFunction>> cne(task, seedGenome, params);

  // Evolve.
  cne.Evolve();  // Judge whether XOR test passed or not by printing 
                 // the best fitness during each generation.
}

BOOST_AUTO_TEST_SUITE_END();
