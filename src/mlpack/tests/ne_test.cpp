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
                             fitness);

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
  params.aSpeciesSize = 500;
  params.aMutateRate = 0.1;
  params.aMutateSize = 0.02;
  params.aElitePercentage = 0.2;
  params.aMaxGeneration = 1000;

  // Construct seed genome for xor task.
  ssize_t id = 0;
  ssize_t numInput = 3;
  ssize_t numOutput = 1;
  double fitness = -1;
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
                             fitness);

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
  params.aMutateAddForwardLinkProb = 0.9;
  params.aMutateAddBackwardLinkProb = 0;
  params.aMutateAddRecurrentLinkProb = 0;
  params.aMutateAddBiasLinkProb = 0;
  params.aMutateAddNeuronProb = 0.6;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;
  params.aNumSpeciesThreshold = 10;

  // Construct seed genome for xor task.
  ssize_t id = 0;
  ssize_t numInput = 3;
  ssize_t numOutput = 1;
  double fitness = -1;
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
  LinkGene link2(1, 3, 1, 0, true);
  LinkGene link3(2, 3, 2, 0, true);
  LinkGene link4(0, 4, 3, 0, true);
  LinkGene link5(1, 4, 4, 0, true);
  LinkGene link6(2, 4, 5, 0, true);
  LinkGene link7(4, 3, 6, 0, true);

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
                             fitness);

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
  ssize_t num_trial = 10;
  ssize_t num_step = std::pow(10, 5);

  // Construct task instance.
  TaskCartPole task(mc, mp, g, l, F, tau, track_limit, theta_limit, num_trial, num_step, false);

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
  params.aMutateAddForwardLinkProb = 0.8;
  params.aMutateAddBackwardLinkProb = 0;
  params.aMutateAddRecurrentLinkProb = 0;
  params.aMutateAddBiasLinkProb = 0;
  params.aMutateAddNeuronProb = 0.1;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;
  params.aNumSpeciesThreshold = 10;

  // Set seed genome for cart pole task.
  ssize_t id = 0;
  ssize_t numInput = 5;
  ssize_t numOutput = 1;
  double fitness = -1;
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

  LinkGene link1(0, 6, 0, 0, true);
  LinkGene link2(1, 6, 1, 0, true);
  LinkGene link3(2, 6, 2, 0, true);
  LinkGene link4(3, 6, 3, 0, true);
  LinkGene link5(4, 6, 4, 0, true);
  LinkGene link6(6, 5, 5, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness);

  // Construct NEAT instance.
  NEAT<TaskCartPole> neat(task, seedGenome, params);

  // Evolve. 
  neat.Evolve();
}

/**
 * Test NEAT by Markov Double Pole task.
 */
BOOST_AUTO_TEST_CASE(NENeatMarkovDoublePoleTest)
{
  mlpack::math::RandomSeed(1);

  // Set parameters of cart pole task.
  bool markov = true;

  // Construct task instance.
  TaskDoublePole task(markov);

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
  params.aMutateAddForwardLinkProb = 0.8;
  params.aMutateAddBackwardLinkProb = 0;
  params.aMutateAddRecurrentLinkProb = 0;
  params.aMutateAddBiasLinkProb = 0;
  params.aMutateAddNeuronProb = 0.1;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;
  params.aNumSpeciesThreshold = 10;

  // Set seed genome for Markov double pole task.
  ssize_t id = 0;
  ssize_t numInput = 7;
  ssize_t numOutput = 1;
  double fitness = DBL_MAX;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene3(2, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene4(3, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene5(4, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene6(5, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(6, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene(7, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(8, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(inputGene3);
  neuronGenes.push_back(inputGene4);
  neuronGenes.push_back(inputGene5);
  neuronGenes.push_back(inputGene6);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 8, 0, 0, true);
  LinkGene link2(1, 8, 1, 0, true);
  LinkGene link3(2, 8, 2, 0, true);
  LinkGene link4(3, 8, 3, 0, true);
  LinkGene link5(4, 8, 4, 0, true);
  LinkGene link6(5, 8, 5, 0, true);
  LinkGene link7(6, 8, 6, 0, true);
  LinkGene link8(8, 7, 7, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);
  linkGenes.push_back(link7);
  linkGenes.push_back(link8);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness);

  // Construct NEAT instance.
  NEAT<TaskDoublePole> neat(task, seedGenome, params);

  // Evolve. 
  neat.Evolve();
}

/**
 * Test NEAT by Non-Markov Double Pole task.
 */
BOOST_AUTO_TEST_CASE(NENeatNonMarkovDoublePoleTest)
{
  mlpack::math::RandomSeed(1);

  // Set parameters of cart pole task.
  bool markov = false;

  // Construct task instance.
  TaskDoublePole task(markov);

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
  params.aMutateAddForwardLinkProb = 2.0;
  params.aMutateAddBackwardLinkProb = 0.5;
  params.aMutateAddRecurrentLinkProb = 0.5;
  params.aMutateAddBiasLinkProb = 0.5;
  params.aMutateAddNeuronProb = 0.1;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;
  params.aNumSpeciesThreshold = 10;

  // Set seed genome for Markov double pole task.
  ssize_t id = 0;
  ssize_t numInput = 4;
  ssize_t numOutput = 1;
  double fitness = DBL_MAX;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene3(2, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(3, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene(4, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(5, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(inputGene3);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 5, 0, 0, true);
  LinkGene link2(1, 5, 1, 0, true);
  LinkGene link3(2, 5, 2, 0, true);
  LinkGene link4(3, 5, 3, 0, true);
  LinkGene link5(5, 4, 4, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness);

  // Construct NEAT instance.
  NEAT<TaskDoublePole> neat(task, seedGenome, params);

  // Evolve. 
  neat.Evolve();
}

/**
 * Test NEAT by Mountain Car task.
 */
BOOST_AUTO_TEST_CASE(NENeatMountainCarTest)
{
  mlpack::math::RandomSeed(1);

  // Set parameters of cart pole task.
  double x_l = -1.2;
  double x_h = 0.5;
  double x_dot_l = -0.07;
  double x_dot_h = 0.07;
  double gravity = -0.0025;
  double goal = 0.5;
  ssize_t num_trial = 10;
  ssize_t num_step = std::pow(10, 2);

  // Construct task instance.
  TaskMountainCar task(x_l, x_h, x_dot_l, x_dot_h, gravity, goal, num_trial, num_step, false);

  // Set parameters of NEAT algorithm.
  Parameters params;
  params.aPopulationSize = 50;
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
  params.aMutateAddForwardLinkProb = 0.8;
  params.aMutateAddBackwardLinkProb = 0;
  params.aMutateAddRecurrentLinkProb = 0;
  params.aMutateAddBiasLinkProb = 0;
  params.aMutateAddNeuronProb = 0.1;
  params.aMutateEnabledProb = 0.2;
  params.aMutateDisabledProb = 0.2;
  params.aNumSpeciesThreshold = 10;

  // Set seed genome for cart pole task.
  ssize_t id = 0;
  ssize_t numInput = 3;
  ssize_t numOutput = 3;
  double fitness = -1;
  std::vector<NeuronGene> neuronGenes;
  std::vector<LinkGene> linkGenes;

  NeuronGene inputGene1(0, INPUT, LINEAR, 0, 0, 0);
  NeuronGene inputGene2(1, INPUT, LINEAR, 0, 0, 0);
  NeuronGene biasGene(2, BIAS, LINEAR, 0, 0, 0);
  NeuronGene outputGene1(3, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene outputGene2(4, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene outputGene3(5, OUTPUT, SIGMOID, 1, 0, 0);
  NeuronGene hiddenGene(6, HIDDEN, SIGMOID, 0.5, 0, 0);

  neuronGenes.push_back(inputGene1);
  neuronGenes.push_back(inputGene2);
  neuronGenes.push_back(biasGene);
  neuronGenes.push_back(outputGene1);
  neuronGenes.push_back(outputGene2);
  neuronGenes.push_back(outputGene3);
  neuronGenes.push_back(hiddenGene);

  LinkGene link1(0, 6, 0, 0, true);
  LinkGene link2(1, 6, 1, 0, true);
  LinkGene link3(2, 6, 2, 0, true);
  LinkGene link4(6, 3, 3, 0, true);
  LinkGene link5(6, 4, 4, 0, true);
  LinkGene link6(6, 5, 5, 0, true);

  linkGenes.push_back(link1);
  linkGenes.push_back(link2);
  linkGenes.push_back(link3);
  linkGenes.push_back(link4);
  linkGenes.push_back(link5);
  linkGenes.push_back(link6);

  Genome seedGenome = Genome(0, 
                             neuronGenes,
                             linkGenes,
                             numInput,
                             numOutput,
                             fitness);

  // Construct NEAT instance.
  NEAT<TaskMountainCar> neat(task, seedGenome, params);

  // Evolve. 
  neat.Evolve();
}

BOOST_AUTO_TEST_SUITE_END();
