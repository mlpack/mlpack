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
template<typename Model, typename InitializationRuleType, class Noise>
GAN<Model, InitializationRuleType, Noise>::GAN(
    arma::mat& predictors,
    Model& generator,
    Model& discriminator,
    InitializationRuleType initializeRule,
    Noise noiseFunction,
    size_t noiseDim,
    size_t batchSize,
    size_t generatorUpdateStep,
    size_t preTrainSize):
    predictors(predictors),
    generator(generator),
    discriminator(discriminator),
    initializeRule(initializeRule),
    noiseFunction(noiseFunction),
    noiseDim(noiseDim),
    batchSize(batchSize),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    reset(false)
{
  // Insert IdentityLayer for Joing the generator and discriminator
  discriminator.network.insert(
      discriminator.network.begin(), 
      new IdentityLayer<>());

  counter = 0;
  currentBatch = 0;

  discriminator.deterministic = generator.deterministic = true;

  responses.set_size(1, predictors.n_cols);
  responses.ones();

  discriminator.predictors.set_size(predictors.n_rows, predictors.n_cols + 1);
  discriminator.predictors.cols(0, predictors.n_cols - 1) = predictors;

  discriminator.responses.set_size(1, predictors.n_cols + 1);
  discriminator.responses.ones();
  discriminator.responses(predictors.n_cols) = 0;

  numFunctions = predictors.n_cols;

  noise.set_size(noiseDim, 1);
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::Reset()
{
  size_t genWeights = 0;
  size_t discWeights = 0;

  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);

  for (size_t i = 0; i < generator.network.size(); ++i)
    genWeights += boost::apply_visitor(weightSizeVisitor, generator.network[i]);

  for (size_t i = 0; i < discriminator.network.size(); ++i)
    discWeights += boost::apply_visitor(weightSizeVisitor, 
        discriminator.network[i]);

  parameter.set_size(genWeights + discWeights, 1);
  generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
      false);
  discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
      discWeights, 1 , false, false);

  // Intialise the parmaeters generator
  networkInit.Initialize(generator.network, parameter);
  // Intialise the parameters discriminator
  networkInit.Initialize(discriminator.network, parameter, genWeights);

  reset = true;
}



template<typename Model, typename InitializationRuleType, typename Noise>
template<typename OptimizerType>
void GAN<Model, InitializationRuleType, Noise>::Train(
    OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  Optimizer.Optimize(*this, parameter);
}

template<typename Model, typename InitializationRuleType, typename Noise>
double GAN<Model, InitializationRuleType, Noise>::Evaluate(
    const arma::mat& /*parameters*/,
    const size_t i)
{
  if (!reset)
    Reset();

  currentInput = this->predictors.unsafe_col(i);
  currentTarget = this->responses.unsafe_col(i);
  discriminator.Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
          outputParameterVisitor, 
          discriminator.network.back())), std::move(currentTarget));
  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  arma::mat temp = boost::apply_visitor(
      outputParameterVisitor, generator.network.back());
  temp.reshape(discriminator.predictors.n_rows, 1);
  discriminator.predictors.col(numFunctions) = temp;
  discriminator.Forward(std::move(discriminator.predictors.col(numFunctions)));
  discriminator.responses(numFunctions) = 0;

  currentTarget = discriminator.responses.unsafe_col(numFunctions);
  res += discriminator.outputLayer.Forward(
        std::move(boost::apply_visitor(
            outputParameterVisitor, 
            discriminator.network.back())), std::move(currentTarget));
  return res;
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::
Gradient(const arma::mat& /*parameters*/, const size_t i, arma::mat& gradient)
{
  if (!reset)
    Reset();

  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      Reset();
    }
    gradient = arma::zeros<arma::mat>(parameter.n_elem, 1);
  }
  else
  {
    gradient.zeros();
  }

  if (noiseGradientDiscriminator.is_empty())
  {
    noiseGradientDiscriminator = arma::zeros<arma::mat>(
        gradientDiscriminator.n_elem, 1);
  }
  else
  {
    noiseGradientDiscriminator.zeros();
  }

  gradientGenerator = arma::mat(gradient.memptr(),
      generator.Parameters().n_elem, 1, false, false);

  gradientDiscriminator = arma::mat(gradient.memptr() + 
      gradientGenerator.n_elem,
      discriminator.Parameters().n_elem, 1, false, false);

  // get the gradients of the discriminator
  discriminator.Gradient(discriminator.parameter, i, gradientDiscriminator);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  discriminator.predictors.col(numFunctions) = boost::apply_visitor(
      outputParameterVisitor, generator.network.back());
  discriminator.responses(numFunctions) = 0;

  discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator);

  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize log(1 - D(G(noise)))
    // pass the error from discriminator to generator
    generator.error = boost::apply_visitor(deltaVisitor,
        discriminator.network[1]);
    generator.currentInput = noise;
    generator.Backward();
    generator.ResetGradients(gradientGenerator);
    // set the current input requrired for gradient computation
    generator.Gradient();

    double multiplier = 1.0;

    gradientGenerator = -gradientGenerator;
    gradientGenerator *= multiplier;
/*
    if (counter % batchSize == 0)
    {
      Log::Info << "gradientDiscriminator = " << std::max(std::fabs(gradientDiscriminator.min()), std::fabs(gradientDiscriminator.max())) << std::endl;
      Log::Info << "gradientGenerator = " << std::max(std::fabs(gradientGenerator.min()), std::fabs(gradientGenerator.max())) << std::endl;
    }
*/
  }
  counter++;
  if (counter >= numFunctions)
  {
    counter = 0;
    currentBatch++;
  }
  else if (counter % batchSize == 0)
  {
    currentBatch++;
    if (preTrainSize > 0)
    {
      preTrainSize--;
    }
    if (preTrainSize)
      Log::Info << "Pre Training batch " << preTrainSize << std::endl;
  }
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::Forward(arma::mat&& input)
{
  if (!reset)
    Reset();

  generator.Forward(std::move(input));
  ganOutput = boost::apply_visitor(
      outputParameterVisitor,
      generator.network.back());
  discriminator.Forward(std::move(ganOutput));
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::
Predict(arma::mat&& input, arma::mat& output)
{
  if (!reset)
    Reset();

  Forward(std::move(input));

  output = boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back());
}

} // namespace ann
} // namespace mlpack
# endif
