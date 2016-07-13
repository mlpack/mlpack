/**
 * @file ne_test.cpp
 * @author Bang Liu
 *
 * Test file for NE (Neural Evolution).
 */
#include <cstddef>
#include <cmath>

#include <mlpack/core.hpp>
#include <mlpack/methods/ne/utils.hpp>
#include <mlpack/methods/ne/parameters.hpp>
#include <mlpack/methods/ne/tasks.hpp>
#include <mlpack/methods/ne/link_gene.hpp>
#include <mlpack/methods/ne/neuron_gene.hpp>
#include <mlpack/methods/ne/genome.hpp>
#include <mlpack/methods/ne/species.hpp>
#include <mlpack/methods/ne/cne.hpp>
#include <mlpack/methods/ne/neat.hpp>

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
  LinkGene linkGene(1, 2, 0, 10, true);

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
  LinkGene linkGene3(1, 3, 1, 7, true);
  linkGene = linkGene3;
  BOOST_REQUIRE_EQUAL(linkGene3.InnovationId(), 1);

}

/**
 * Test NeuronGene class.
 */
BOOST_AUTO_TEST_CASE(NENeuronGeneTest)
{
  // Create a neuron gene.
  NeuronGene neuronGene(1, INPUT, SIGMOID, 0, 0, 0);

  // Test parametric constructor and access functions.
  BOOST_REQUIRE_EQUAL(neuronGene.Id(), 1);
  BOOST_REQUIRE_EQUAL(neuronGene.Type(), INPUT);
  BOOST_REQUIRE_EQUAL(neuronGene.ActFuncType(), SIGMOID);
  BOOST_REQUIRE_EQUAL(neuronGene.Input(), 0);
  BOOST_REQUIRE_EQUAL(neuronGene.Activation(), 0);

  // Test set function
  neuronGene.ActFuncType(RELU);
  BOOST_REQUIRE_EQUAL(neuronGene.ActFuncType(), RELU);

  // Test copy function.
  NeuronGene neuronGene2(neuronGene);
  BOOST_REQUIRE_EQUAL(neuronGene2.Id(), 1);

  // Test operator =.
  NeuronGene neuronGene3(11, INPUT, SIGMOID, 0, 0, 0);
  neuronGene = neuronGene3;
  BOOST_REQUIRE_EQUAL(neuronGene.Id(), 11);

}

/**
 * Test Genome class.
 */
BOOST_AUTO_TEST_CASE(NEGenomeTest)
{
  mlpack::math::RandomSeed(1);

  // Construct seed genome for xor task.
  ssize_t id = 0;
  ssize_t numInput = 3;
  ssize_t numOutput = 1;
  double fitness = -1;
  double adjustedFitness = -1;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(2, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene(3, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(4, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 3, 0, 1, true);
  LinkGene link2(1, 3, 0, 1, true);
  LinkGene link3(2, 3, 0, 0.5, true);
  LinkGene link4(0, 4, 0, -0.5, true);
  LinkGene link5(1, 4, 0, 2, true);
  LinkGene link6(2, 4, 0, 1, true);
  LinkGene link7(4, 3, 0, 0.1, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);
  linkGenes.push_back(link7);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness,
                             adjustedFitness);

  // Test seed genome.
  std::vector<std::vector<double>> inputs;  // TODO: use arma::mat for input.
  std::vector<double> input1 = {0, 0, 1};
  std::vector<double> input2 = {0, 1, 1};
  std::vector<double> input3 = {1, 0, 1};
  std::vector<double> input4 = {1, 1, 1};
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<double> outputs;
  outputs.push_back(1);
  outputs.push_back(0);
  outputs.push_back(1);
  outputs.push_back(0);

  for (int i=0; i<4; ++i) {
    seedGenome.Activate(inputs[i]);
    std::vector<double> output;
    seedGenome.Output(output);
    std::cout << "Output is: " << output[0] << std::endl;
  }
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
  params.aSpeciesSize = 1000;
  params.aMutateRate = 0.1;
  params.aMutateSize = 0.02;
  params.aElitePercentage = 0.2;
  params.aMaxGeneration = 500;

  // Construct seed genome for xor task.
  ssize_t id = 0;
  ssize_t numInput = 3;
  ssize_t numOutput = 1;
  double fitness = -1;
  double adjustedFitness = -1;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(2, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene(3, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(4, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 3, 0, 0, true);
  LinkGene link2(1, 3, 0, 0, true);
  LinkGene link3(2, 3, 0, 0, true);
  LinkGene link4(0, 4, 0, 0, true);
  LinkGene link5(1, 4, 0, 0, true);
  LinkGene link6(2, 4, 0, 0, true);
  LinkGene link7(4, 3, 0, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);
  linkGenes.push_back(link7);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness,
                             adjustedFitness);

  // Specify task type.
  TaskXor<ann::MeanSquaredErrorFunction> task;

  // Construct CNE instance.
  CNE<TaskXor<ann::MeanSquaredErrorFunction>> cne(task, seedGenome, params);

  // Evolve.
  cne.Evolve();  // Judge whether XOR test passed or not by printing 
                 // the best fitness during each generation.
}

/**
 * Test CNE by XOR task.
 */
BOOST_AUTO_TEST_CASE(NENeatXorTest)
{
  mlpack::math::RandomSeed(1);

  // Set NEAT algorithm parameters.
  Parameters params;
  params.aPopulationSize = 500;
  params.aMaxGeneration = 500;
  params.aCoeffDisjoint = 2.0;
  params.aCoeffWeightDiff = 0.4;
  params.aCompatThreshold = 1.0;
  params.aStaleAgeThreshold = 15;
  params.aCrossoverRate = 0.75;
  params.aCullSpeciesPercentage = 0.5;
  params.aMutateWeightProb = 0.2;
  params.aPerturbWeightProb = 0.9;
  params.aMutateWeightSize = 0.1;
  params.aMutateAddLinkProb = 0.5;
  params.aMutateAddRecurrentLinkProb = 0;
  params.aMutateAddLoopLinkProb = 0;
  params.aMutateAddNeuronProb = 0.5;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;

  // Construct seed genome for xor task.
  ssize_t id = 0;
  ssize_t numInput = 3;
  ssize_t numOutput = 1;
  double fitness = -1;
  double adjustedFitness = -1;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(2, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene(3, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(4, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 3, 0, 0, true);
  LinkGene link2(1, 3, 0, 0, true);
  LinkGene link3(2, 3, 0, 0, true);
  LinkGene link4(0, 4, 0, 0, true);
  LinkGene link5(1, 4, 0, 0, true);
  LinkGene link6(2, 4, 0, 0, true);
  LinkGene link7(4, 3, 0, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);
  linkGenes.push_back(link7);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness,
                             adjustedFitness);

  // Specify task type.
  TaskXor<ann::MeanSquaredErrorFunction> task;

  // Construct NEAT instance.
  NEAT<TaskXor<ann::MeanSquaredErrorFunction>> neat(task, seedGenome, params);

  // Evolve.
  neat.Evolve();  // Judge whether XOR test passed or not by printing 
                  // the best fitness during each generation.
}

/**
 * Test NEAT by Cart Pole task.
 */
BOOST_AUTO_TEST_CASE(NENeatCartPoleTest)
{
  mlpack::math::RandomSeed(1);

  // Set parameters of cart pole task.
  double track_limit = 2.4;
  double theta_limit = 12 * M_PI / 180.0;
  double g = 9.81;
  double mp = 0.1;
  double mc = 1.0;
  double l = 0.5;
  double F = 10.0;
  double tau = 0.02;
  ssize_t num_trial = 20;
  ssize_t num_step = 200;

  // Construct task instance.
  TaskCartPole task(mc, mp, g, l, F, tau, track_limit, theta_limit, num_trial, num_step);

  // Set parameters of NEAT algorithm.
  Parameters params;
  params.aPopulationSize = 500;
  params.aMaxGeneration = 500;
  params.aCoeffDisjoint = 2.0;
  params.aCoeffWeightDiff = 0.4;
  params.aCompatThreshold = 1.0;
  params.aStaleAgeThreshold = 15;
  params.aCrossoverRate = 0.75;
  params.aCullSpeciesPercentage = 0.5;
  params.aMutateWeightProb = 0.2;
  params.aPerturbWeightProb = 0.9;
  params.aMutateWeightSize = 0.1;
  params.aMutateAddLinkProb = 0.5;
  params.aMutateAddRecurrentLinkProb = 0;
  params.aMutateAddLoopLinkProb = 0;
  params.aMutateAddNeuronProb = 0.5;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;

  // Set seed genome for cart pole task.
  ssize_t id = 0;
  ssize_t numInput = 5;
  ssize_t numOutput = 1;
  double fitness = -1;
  double adjustedFitness = -1;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene3(2, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene4(3, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(4, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene(5, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(6, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(inputGene3);
  neuronGenes.push_back(inputGene4);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 5, 0, 0, true);
  LinkGene link2(1, 5, 0, 0, true);
  LinkGene link3(2, 5, 0, 0, true);
  LinkGene link4(3, 5, 0, 0, true);
  LinkGene link5(4, 5, 0, 0, true);
  LinkGene link6(0, 6, 0, 0, true);
  LinkGene link7(1, 6, 0, 0, true);
  LinkGene link8(2, 6, 0, 0, true);
  LinkGene link9(3, 6, 0, 0, true);
  LinkGene link10(4, 6, 0, 0, true);
  LinkGene link11(6, 5, 0, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);
  linkGenes.push_back(link7);
  linkGenes.push_back(link8);
  linkGenes.push_back(link9);
  linkGenes.push_back(link10);
  linkGenes.push_back(link11);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness,
                             adjustedFitness);

  // Construct NEAT instance.
  NEAT<TaskCartPole> neat(task, seedGenome, params);

  // Evolve. 
  neat.Evolve();  // Fitness 4000 (num_trial * num_step) means passed test.
}

BOOST_AUTO_TEST_SUITE_END();
