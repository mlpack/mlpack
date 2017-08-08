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
#include <mlpack/methods/ann/init_rules/network_init.hpp>
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
template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>
::GenerativeAdversarialNetwork(arma::mat& trainData,
    InitializationRuleType initializeRule,
    Generator& generator,
    Discriminator& discriminator,
    size_t batchSize,
    size_t noiseInSize,
    size_t disIteration):
    trainData(trainData),
    initializeRule(initializeRule),
    generator(generator),
    discriminator(discriminator),
    trainGenerator(false),
    batchSize(batchSize),
    noiseInSize(noiseInSize),
    disIteration(2 * disIteration),
    reset(false)
{
  discriminator.network.insert(discriminator.network.begin(),
      new IdentityLayer<>());
  trainGenerator = false;
  iterations = 1;
  offset = 0;
  iterationDiscriminator = disIteration;
}

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>
::Reset()
{
  std::cout << "Reset function" << std::endl;
  size_t generatorWeights = 0;
  size_t discriminatorWeights = 0;

  for (size_t i = 0; i < generator.network.size(); ++i)
    generatorWeights += boost::apply_visitor(weightSizeVisitor,
        generator.network[i]);

  assert(generatorWeights > 0);

  for (size_t i = 0; i < discriminator.network.size(); ++i)
    discriminatorWeights += boost::apply_visitor(weightSizeVisitor,
        discriminator.network[i]);

  assert(discriminatorWeights > 0);

  parameter.set_size(generatorWeights + discriminatorWeights, 1);

  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  // Intialise the parmaeters generator
  networkInit.Initialize(generator.network, parameter);
  // Intialise the parameters discriminator
  networkInit.Initialize(discriminator.network, parameter, generatorWeights);

  reset = true;
}

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>
::GenerateData()
{
  responses.set_size(1, batchSize);
  discriminator.responses = arma::mat(responses.memptr(), 1, batchSize,
        false, false);
  size_t n_rows = trainData.n_rows;
  size_t disBatchSize = std::floor(batchSize / 2.0);

  if (!trainGenerator)
  {
    predictors.set_size(trainData.n_rows, batchSize);
    noiseData.set_size(noiseInSize, disBatchSize);
    
    disFakeData = arma::mat(predictors.memptr(), n_rows, disBatchSize,
        false, false);
    
    predictors.cols(disBatchSize, batchSize - 1) =arma::mat(
        trainData.memptr() + offset, n_rows, batchSize - disBatchSize,
        true, false);

    discriminator.predictors = arma::mat(predictors.memptr(), n_rows, batchSize,
        false, false);

    GenerateNoise(std::move(disFakeData), std::move(noiseData.cols(0,
        disBatchSize - 1)));

    responses.cols(0, disBatchSize - 1).zeros();
    responses.cols(disBatchSize, batchSize - 1).ones();

    this->numFunctions = predictors.n_cols;
    discriminator.numFunctions = predictors.n_cols;
  }
  else
  {
    predictors.set_size(noiseInSize, batchSize);
    noiseData.set_size(noiseInSize, batchSize);
    
    noiseData = arma::mat(predictors.memptr(), noiseInSize,
        batchSize, false, false);
    genFakeData = arma::mat(discriminator.predictors.memptr(), n_rows,
        batchSize, false, false);
    GenerateNoise(std::move(genFakeData), std::move(noiseData));
    
    responses.ones();
    this->numFunctions = predictors.n_cols;
  }
}

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
template<typename OptimizerType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>
::Train(OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  offset = 0;
  trainGenerator = false;
  GenerateData();
  Optimizer.Optimize(*this, parameter);
}

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
double GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>::Evaluate(const arma::mat& /*parameters*/,
                                      const size_t i,
                                      const bool /*deterministic*/)
{
  currentInput = this->predictors.unsafe_col(i);
  currentTarget = this->responses.unsafe_col(i);
  Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(std::move(
      boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget));
  return res;
}

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>::Gradient(const arma::mat& parameters,
                                      const size_t i, arma::mat& gradient)
{
  if (!reset)
    Reset();

  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      Reset();
    }
    gradient = arma::zeros<arma::mat>(parameter.n_rows, 1);
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
    // set the current input requrired for gradient computation
    generator.currentInput = predictors.col(i);
    generator.Gradient();
  }

  // Freeze weights of discrminator if generator is training
  if (!trainGenerator)
    gradientGenerator.zeros();
  else
    gradientDiscriminator.zeros();

  // if full batch seen then alternate training
  if (iterations % numFunctions == 0)
  {
    if (!trainGenerator && iterationDiscriminator==1)
    {
      trainGenerator = true;
      GenerateData();
      iterationDiscriminator = 2 * disIteration;
    }
    else
    {
      trainGenerator = false;
      offset = (offset + batchSize) % trainData.n_cols;
      GenerateData();
      iterationDiscriminator--;
    }
  }
  iterations++;
  std::cout << "parameters generator" << std::endl;
  parameter.rows(0, 10).print();
  std::cout << "parameters discriminator" << std::endl;
  parameter.rows(generator.Parameters().n_rows,
      generator.Parameters().n_rows + 10).print();
}

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>::Forward(arma::mat&& input)
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

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>::Predict(arma::mat&& input, arma::mat& output)
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

template<typename Generator,
         typename Discriminator,
         typename InitializationRuleType>
void GenerativeAdversarialNetwork<Generator,
    Discriminator,
    InitializationRuleType>
::GenerateNoise(arma::mat&& fakeData,
                arma::mat&& noiseData)
{
  if (!reset)
    Reset();

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
