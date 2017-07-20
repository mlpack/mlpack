/**
 * @file gan.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_IMPL_HPP

#include "gan.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>


#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::math;
using namespace mlpack::distribution;

namespace mlpack {
namespace ann /** Restricted Boltzmann Machine.  */ {
template<typename Generator, typename Discriminator, typename IntializerType>
GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::GenerativeAdversarialNetwork(arma::mat trainData,
    arma::mat trainLables,
    IntializerType initializeRule,
    Generator& generator,
    Discriminator& discriminator,
    size_t batchSize,
    size_t generatorInSize):
    initializeRule(initializeRule),
    generator(generator),
    discriminator(discriminator),
    trainGenerator(false),
    batchSize(batchSize),
    generatorInSize(generatorInSize),
    reset(false)
{
  numFunctions = trainData.n_cols;
  predictors = trainData;
  responses = trainLables;
  fakeData.set_size(trainData.n_rows, batchSize);
}
template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Reset()
{
  // Call the reset function of both the Generator and Discriminator Network
  generator.ResetParameters();
  discriminator.ResetParameters();
  fakeData.set_size(predictors.n_rows, batchSize);
  fakeLables.set_size(1, batchSize);
  tempTrainData.set_size(predictors.n_rows, 2 * batchSize);
  tempLabels.set_size(1, 2 * batchSize);
  fakeData.zeros();
  fakeLables.zeros();
  parameter.set_size(generator.Parameters().n_rows + 
      discriminator.Parameters().n_rows,
      generator.Parameters().n_cols + generator.Parameters().n_cols);
  initializeRule.Initialize(parameter, parameter.n_elem, 1);
  generator.Parameters() = arma::mat(parameter.memptr(),
      generator.Parameters().n_rows, generator.Parameters().n_cols,
      false, false);
  discriminator.Parameters() = arma::mat(parameter.memptr(),
    discriminator.Parameters().n_rows, discriminator.Parameters().n_cols,
    false, false);
  reset = true;
}
template<typename Generator, typename Discriminator, typename IntializerType>
template<typename OptimizerType>
void GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Train(OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  size_t offset = 0;
  for (size_t i = 0; i< 1; i++)
  {
    std::cout << " training epoch =" << i << std::endl;
    // Generate fake data
    Generate(batchSize, std::move(fakeData));
    std::cout << "fake Data size = " << arma::size(fakeData) << std::endl;

    // Create training data for discrminator
    tempTrainData.cols(0, batchSize - 1) = arma::mat(
      fakeData.memptr() + offset, fakeData.n_rows, batchSize, false, false);
    tempTrainData.cols(batchSize, tempTrainData.n_cols - 1) = arma::mat(
        predictors.memptr() + offset, predictors.n_rows, batchSize,
        false, false);
    tempLabels.cols(0, batchSize - 1) = arma::mat(
        fakeLables.memptr() + offset, fakeLables.n_rows, batchSize,
        false, false);
    tempLabels.cols(batchSize, tempLabels.n_cols - 1) = arma::mat(
        responses.memptr() + offset,responses.n_rows, batchSize,
        false, false);
    offset += batchSize;

    // Train the discrminator network
    this->predictors = std::move(tempTrainData);
    this->responses = std::move(tempLabels);
    numFunctions = predictors.n_cols;
    discriminator.predictors = predictors;
    discriminator.responses = responses;
    generator.predictors = predictors;
    generator.responses = responses;
    std::cout << arma::size(discriminator.predictors) << std::endl;
    std::cout << arma::size(predictors) << std::endl;
    Optimizer.Optimize(*this, parameter);
    trainGenerator = true;
    // Train the generator network
    Generate(predictors.n_cols, std::move(fakeData));
    this->predictors = fakeData;
    this->responses = arma::ones(1, responses.n_cols);
    discriminator.predictors = predictors;
    discriminator.responses = responses;
    numFunctions = predictors.n_cols;
    std::cout << "predictors.n_cols" << predictors.n_cols << std::endl;
    std::cout << "NumFunctions = " << numFunctions << std::endl;
    Optimizer.MaxIterations() *= 10;
    Optimizer.Optimize(*this, parameter);
    // set train generator to false
    trainGenerator = false;
  }
}
template<typename Generator, typename Discriminator, typename IntializerType>
double GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Evaluate(const arma::mat& /*parameters*/,
    const size_t i, const bool /*deterministic*/)
{
  /*
  arma::mat currentInput = predictors.unsafe_col(i);
  arma::mat currentTarget = responses.unsafe_col(i);
  discriminator.Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(std::move(boost::apply_visitor(
      outputParameterVisitor, discriminator.network.back())),
      std::move(currentTarget));
  std::cout << res << std::endl;
  */
  return 0;
}

/**
 * Gradient function 
 */
template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Gradient(const arma::mat& parameters, const size_t i, arma::mat& gradient)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      Reset();
    }
  gradient = arma::zeros<arma::mat>(generator.Parameters().n_rows +
      discriminator.Parameters().n_rows, generator.Parameters().n_cols +
      discriminator.Parameters().n_rows);
  }
  else
  {
    gradient.zeros();
  }

  // Gradient for generator network
  gradientGenerator = arma::mat(gradient.memptr(),
      generator.Parameters().n_rows,
      generator.Parameters().n_cols, false, false);

  // Gradient for discriminator network
  gradientDisriminator = arma::mat(gradient.memptr(),
      discriminator.Parameters().n_rows,
      discriminator.Parameters().n_cols, false, false);

  // Get the discriminator gradient 
  discriminator.Gradient(parameters, i, gradientDisriminator);
  if (trainGenerator)
  {
    // Use visitors later
    boost::apply_visitor(DeltaVisitor(), discriminator.network.front()) =
        generator.outputLayer.Delta();
    generator.Gradient(parameters, i, gradientGenerator);
  }
  // Freeze weights of discrminator if generator is training
  if (!trainGenerator)
    gradientGenerator.zeros();
  else
    gradientDisriminator.zeros();
}

/**
 * This function does Forward pass through the GAN
 * network.
 *
 * @param input  the noise input
 */
template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Forward(arma::mat&& input)
{
  if (!reset)
    Reset();
  generator.Forward(std::move(input));
  ganOutput = boost::apply_visitor(outputParameterVisitor,
      generator.network.back());
  discriminator.Forward(std::move(ganOutput));
}

/**
 * This function predicts the output of the network
 * on the given input
 *
 * @param input  the input  the discriminator network
 * @param output result of the discriminator network
 */
template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Predict(arma::mat&& input, arma::mat& output)
{
  if (!reset)
    Reset();
  discriminator.Forward(std::move(input));
  output = boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back());
}

/**
 * Generate function generates random noise 
 * samples from a given distribution with 
 * given args. Samples are stored in a local variable.
 *
 * @param numSamples number of samples to be generated from the distribution
 * @param fakeData fake data to generate (generatorOutSize * numSamples)
 */
template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator,IntializerType>
::Generate(size_t numSamples, arma::mat&& fakeData)
{
  if (!reset)
    Reset();
  arma::mat noise(generatorInSize, 1), temp;
  for (size_t i = 0; i < numSamples; i++)
  {
    for (size_t j = 0; j < generatorInSize; j++)
      noise.row(j) = RandNormal();
    generator.Forward(std::move(noise));
    fakeData.col(i) = boost::apply_visitor(outputParameterVisitor,
          generator.network.back());
  }
}
} // namespace ann
} // namespace mlpack
# endif
