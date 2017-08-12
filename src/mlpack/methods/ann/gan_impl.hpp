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
template<typename Model, typename InitializationRuleType>
GAN<Model, InitializationRuleType>::GAN(
    arma::mat& predictors,
    Model& generator,
    Model& discriminator,
    InitializationRuleType initializeRule,
    size_t noiseDim,
    size_t batchSize,
    size_t generatorUpdateStep):
    predictors(predictors),
    generator(generator),
    discriminator(discriminator),
    initializeRule(initializeRule),
    batchSize(batchSize),
    generatorUpdateStep(generatorUpdateStep),
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

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::Reset()
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



template<typename Model, typename InitializationRuleType>
template<typename OptimizerType>
void GAN<Model, InitializationRuleType>::Train(OptimizerType& Optimizer)
{
  if (!reset)
    Reset();

  Optimizer.Optimize(*this, parameter);
}

template<typename Model, typename InitializationRuleType>
double GAN<Model, InitializationRuleType>::Evaluate(
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

  noise.randn();
  generator.Forward(std::move(noise));
  discriminator.predictors.col(numFunctions) = boost::apply_visitor(
      outputParameterVisitor, generator.network.back());
  discriminator.Forward(std::move(discriminator.predictors.col(numFunctions)));
  discriminator.responses(numFunctions) = 0;

  currentTarget = discriminator.responses.unsafe_col(numFunctions);
  res += discriminator.outputLayer.Forward(
        std::move(boost::apply_visitor(
            outputParameterVisitor, 
            discriminator.network.back())), std::move(currentTarget));
  return res;
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::
Gradient(const arma::mat& /*parameters*/, const size_t i, arma::mat& gradient)
{
  static bool print = true;
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
    noiseGradientDiscriminator = arma::zeros<arma::mat>(gradientDiscriminator.n_elem, 1);
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

//  noise.randn();
  noise.imbue( [&]() { return math::Random(0, 1); } );
  generator.Forward(std::move(noise));
  discriminator.predictors.col(numFunctions) = boost::apply_visitor(
      outputParameterVisitor, generator.network.back());
  discriminator.responses(numFunctions) = 0;

  discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator);

  gradientDiscriminator += noiseGradientDiscriminator;

//  size_t numBatches = counter / batchSize;
//  size_t currentBatchSize = (batchSize * (numBatches + 1) <= numFunctions ?
//      batchSize : numFunctions - batchSize * numBatches);

//  if (currentBatch % generatorUpdateStep == 0 && /*counter % batchSize == 0 &&*/ currentBatch != 0)
/*  {
      // Minimize -log(D(G(noise)))


    discriminator.responses(numFunctions) = 1;
    discriminator.Gradient(discriminator.parameter, numFunctions,
        noiseGradientDiscriminator);

    // pass the error from discriminator to generator
    generator.error = boost::apply_visitor(deltaVisitor,
        discriminator.network[1]);
    generator.currentInput = noise;
    generator.Backward();
    generator.ResetGradients(gradientGenerator);
    // set the current input requrired for gradient computation
    generator.Gradient();

//    gradientGenerator *= currentBatchSize;

    if (counter % batchSize == 0 && print)
    {
      Log::Info << "gradientDiscriminator = " << std::max(std::fabs(gradientDiscriminator.min()), std::fabs(gradientDiscriminator.max())) << std::endl;
      Log::Info << "gradientGenerator = " << std::max(std::fabs(gradientGenerator.min()), std::fabs(gradientGenerator.max())) << std::endl;
      print = false;
    }
  }
*/

  if (currentBatch % generatorUpdateStep == 0 &&/* counter % batchSize == 0 &&*/ currentBatch != 0)
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

//    gradientGenerator *= currentBatchSize;

    if (counter % batchSize == 0 && print)
    {
      Log::Info << "gradientDiscriminator = " << std::max(std::fabs(gradientDiscriminator.min()), std::fabs(gradientDiscriminator.max())) << std::endl;
      Log::Info << "gradientGenerator = " << std::max(std::fabs(gradientGenerator.min()), std::fabs(gradientGenerator.max())) << std::endl;
      print = false;
    }
  }

  counter++;

// TODO: I know the code below looks like crutches.
// I think it is better to remove batch handling from GAN.
// For example we could add some functions to the Optimizer API
// in order to obtain the number of current batch.
  if (counter >= numFunctions)
  {
    counter = 0;
    currentBatch++;
    print = true;
  }
  else if (counter % batchSize == 0)
    currentBatch++;
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::Forward(arma::mat&& input)
{
  if (!reset)
    Reset();

  generator.Forward(std::move(input));
  ganOutput = boost::apply_visitor(
      outputParameterVisitor,
      generator.network.back());
  discriminator.Forward(std::move(ganOutput));
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::
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
