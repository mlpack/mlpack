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
#include <assert.h>

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
namespace ann /** artifical neural network  */ {
template<typename Generator, typename Discriminator, typename IntializerType>
GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::GenerativeAdversarialNetwork(arma::mat& trainData,
    IntializerType initializeRule,
    Generator& generator,
    Discriminator& discriminator,
    size_t batchSize,
    size_t iterations,
    size_t diteration,
    size_t generatorInSize):
    trainData(trainData),
    initializeRule(initializeRule),
    generator(generator),
    discriminator(discriminator),
    trainGenerator(false),
    batchSize(batchSize),
    iterations(iterations),
    diteration(diteration),
    generatorInSize(generatorInSize),
    reset(false)
{
  discriminator.network.insert(discriminator.network.begin(), new Join<>());
  numFunctions = trainData.n_cols;
  responses = arma::ones(1, trainData.n_cols);
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Reset()
{
  size_t generatorWeights = 0;
  size_t discriminatorWeights = 0;
  size_t offset = 0;

  for (size_t i = 0; i < generator.network.size(); ++i)
    generatorWeights += boost::apply_visitor(weightSizeVisitor,
        generator.network[i]);

  for (size_t i = 0; i < discriminator.network.size(); ++i)
    discriminatorWeights += boost::apply_visitor(weightSizeVisitor,
        discriminator.network[i]);

  parameter.set_size(generatorWeights + discriminatorWeights, 1);

  // Intialise the parmaeters
  initializeRule.Initialize(parameter, parameter.n_elem, 1);

  generator.Parameters() = arma::mat(parameter.memptr(), generatorWeights,
      1, false, false);

  discriminator.Parameters() = arma::mat(parameter.memptr() + generatorWeights,
      discriminatorWeights, 1, false, false);

  // Reset both the generator and discriminator
  for (size_t i = 0; i < generator.network.size(); ++i)
  {
    offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
        offset), generator.network[i]);

    boost::apply_visitor(resetVisitor, generator.network[i]);
  }

  for (size_t i = 0; i < discriminator.network.size(); ++i)
  {
    offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
        offset), discriminator.network[i]);

    boost::apply_visitor(resetVisitor, discriminator.network[i]);
  }

  reset = true;
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::GenerateData(arma::mat& batchData, arma::mat& batchResponses, size_t offset)
{
  if (diteration * batchSize > trainData.n_cols)
  {
    std::cout << "k value too large" << std::endl;
    return;
  }

  if (!trainGenerator)
  {
    size_t temp =  batchSize * diteration;
    batchData.set_size(trainData.n_rows, 2 * temp);
    batchResponses.set_size(1, 2 * temp);
    noiseData.set_size(generatorInSize, temp);
    dData.set_size(trainData.n_rows, temp);
    Generate(std::move(dData), std::move(noiseData));

    batchData.cols(0, temp - 1) = arma::mat(dData.memptr(),
        trainData.n_rows, temp, false, false);
    batchData.cols(temp, batchData.n_cols - 1) = 
        arma::mat(trainData.memptr() + offset, trainData.n_rows,
        temp, false, false);

    batchResponses.cols(0, temp - 1).zeros();
    batchResponses.cols(temp, batchResponses.n_cols - 1).ones();

    this->predictors = std::move(batchData);
    this->responses = std::move(batchLabels);
    numFunctions = predictors.n_cols;
    discriminator.predictors = this->predictors;
    discriminator.responses = this->responses;
  }
  else
  {
    noiseData.set_size(generatorInSize, batchSize);
    batchData.set_size(trainData.n_rows, batchSize);
    gData.set_size(trainData.n_rows, batchSize);
    batchResponses.set_size(1, batchSize);
    Generate(std::move(gData), std::move(noiseData));
    batchData = std::move(gData);
    batchResponses.ones();
    this->predictors = std::move(noiseData);
    this->responses = std::move(batchResponses);
    numFunctions = predictors.n_cols;
    discriminator.predictors = batchData;
    discriminator.responses = this->responses;
  }
}

template<typename Generator, typename Discriminator, typename IntializerType>
template<typename OptimizerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Train(OptimizerType& Optimizer)
{
  if (!reset)
    Reset();

  for (size_t i = 0; i < iterations; i++)
  {
    size_t offset = 0;
    // last batch
    if (offset + batchSize >  trainData.n_cols)
      offset -=  offset + batchSize - trainData.n_cols;

    // Generate fake data for discrminator to train on.
    GenerateData(batchData, batchLabels, offset);
    offset += batchSize * diteration;
    Optimizer.Optimize(*this, parameter);

    // Training Generator
    trainGenerator = true;
    GenerateData(batchData, batchLabels, 0);
    Optimizer.Optimize(*this, parameter);
    // set train generator to false
    trainGenerator = false;
  }
}

template<typename Generator, typename Discriminator, typename IntializerType>
double GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Evaluate(const arma::mat& /*parameters*/,
    const size_t i, const bool /*deterministic*/)
{
  currentInput = this->predictors.unsafe_col(i);
  currentTarget = this->responses.unsafe_col(i);
  Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(std::move(
      boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget)); 
  return res;
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Gradient(const arma::mat& parameters, const size_t i, arma::mat& gradient)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      Reset();
    }
    gradient = arma::zeros<arma::mat>(generator.Parameters().n_rows +
        discriminator.Parameters().n_rows, 1);
  }
  else
  {
    gradient.zeros();
  }

  // Gradient for generator network
  gradientGenerator = arma::mat(gradient.memptr(),
      generator.Parameters().n_rows, 1, false, false);

  // Gradient for discriminator network
  gradientDiscriminator = arma::mat(gradient.memptr() + gradientGenerator.n_elem,
      discriminator.Parameters().n_rows, 1, false, false);

  // get the gradients of the discriminator
  discriminator.Gradient(parameters, i, gradientDiscriminator);
  if (trainGenerator)
  {
    // pass the error from discriminator to generator
    generator.error = boost::apply_visitor(deltaVisitor,
        discriminator.network[1]);
    generator.Backward();
    generator.ResetGradients(gradientGenerator);
    generator.currentInput = predictors.col(i);
    generator.Gradient();
  }
  std::cout << arma::size(gradient) << std::endl;

  // Freeze weights of discrminator if generator is training
  if (!trainGenerator)
    gradientGenerator.zeros();
  else
    gradientDiscriminator.zeros();
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Forward(arma::mat&& input)
{
  if (!reset)
    Reset();

  if (!trainGenerator)
  {
    discriminator.Forward(std::move(input));
    ganOutput = boost::apply_visitor(outputParameterVisitor,
        discriminator.network.back());
  }
  else
  {
    generator.Forward(std::move(input));
    ganOutput = boost::apply_visitor(outputParameterVisitor,
        generator.network.back());
    discriminator.Forward(std::move(ganOutput));
  }
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Predict(arma::mat&& input, arma::mat& output)
{
  if (!reset)
    Reset();

  if (!trainGenerator)
    discriminator.Forward(std::move(input));
  else
    Forward(std::move(input));

  output = boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back());
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Generate(arma::mat&& fakeData, arma::mat&& noiseData)
{
  if (!reset)
    Reset();
  // fill noiseData
  noiseData.imbue( [&]() { return arma::as_scalar(RandNormal()); });
  for (size_t i = 0; i < noiseData.n_cols; i++)
  {
    generator.Forward(std::move(noiseData.col(i)));
    fakeData.col(i) = boost::apply_visitor(outputParameterVisitor,
        generator.network.back());
  }
}
} // namespace ann
} // namespace mlpack
# endif
