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
#include <boost/test/unit_test.hpp>
#include <mlpack/tests/test_tools.hpp>

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
    arma::mat& trainData,
    arma::mat& noiseData,
    Model& generator,
    Model& discriminator,
    InitializationRuleType initializeRule,
    size_t batchSize,
    size_t disIteration):
    trainData(trainData),
    noiseData(noiseData),
    generator(generator),
    discriminator(discriminator),
    initializeRule(initializeRule),
    trainGenerator(false),
    batchSize(batchSize),
    disIteration(2 * disIteration),
    reset(false)
{
  // Insert IdentityLayer for Joing the generator and discriminator
  discriminator.network.insert(
      discriminator.network.begin(), 
      new IdentityLayer<>());
  trainGenerator = false;
  counter = 1;
  offset = 0;
  iterationDiscriminator = disIteration;
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

  // Intialise the parmaeters generator
  networkInit.Initialize(generator.network, parameter);
  // Intialise the parameters discriminator
  networkInit.Initialize(discriminator.network, parameter, genWeights);
  generator.Parameters() = arma::mat(
      parameter.memptr(), genWeights, 1, false, false);
  discriminator.Parameters() = arma::mat(
      parameter.memptr() + genWeights, discWeights, 1 , false, false);
  reset = true;
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::CreateBatch()
{
  responses.set_size(1, batchSize);
  discriminator.responses = arma::mat(responses.memptr(), 1, batchSize,
        false, false);
  size_t n_rows = trainData.n_rows;
  size_t disBatchSize = std::floor(batchSize / 2.0);
  std::cout << "offset = " << offset << std::endl;

  if (!trainGenerator)
  {
    predictors.set_size(n_rows, batchSize);
    
    predictors.cols(0, disBatchSize - 1) = arma::mat(trainData.memptr() + offset, 
          n_rows, disBatchSize, false, false);

    predictors.cols(disBatchSize, batchSize - 1) =arma::mat(
        noiseData.memptr() + offset, n_rows, batchSize - disBatchSize,
        false, false);

    discriminator.predictors = arma::mat(predictors.memptr(), n_rows, batchSize,
        false, false);

    responses.cols(0, disBatchSize - 1).zeros();
    responses.cols(disBatchSize, batchSize - 1).ones();

    this->numFunctions = predictors.n_cols;
    discriminator.numFunctions = predictors.n_cols;
  }
  else
  {
    predictors.set_size(noiseData.n_rows, batchSize);
    discriminator.predictors.set_size(n_rows, batchSize);

    predictors.cols(0, batchSize - 1) = arma::mat(noiseData.memptr() + offset,
        noiseData.n_rows, batchSize, false, false);

    for (size_t i = 0; i < batchSize; i++)
    {
      generator.Forward(std::move(predictors.col(i)));
      discriminator.predictors.col(i) = boost::apply_visitor(
          outputParameterVisitor, generator.network.back());
    }
    responses.ones();
    this->numFunctions = predictors.n_cols;
  }
}

template<typename Model, typename InitializationRuleType>
template<typename OptimizerType>
void GAN<Model, InitializationRuleType>::Train(OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  offset = 0;
  trainGenerator = false;
  CreateBatch();
  Optimizer.Optimize(*this, parameter);
}

template<typename Model, typename InitializationRuleType>
double GAN<Model, InitializationRuleType>::Evaluate(
    const arma::mat& /*parameters*/,
    const size_t i)
{
  currentInput = this->predictors.unsafe_col(i);
  currentTarget = this->responses.unsafe_col(i);
  Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
          outputParameterVisitor, 
          discriminator.network.back())), std::move(currentTarget));
  return res;
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::
Gradient(const arma::mat& parameters, const size_t i, arma::mat& gradient)
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
  gradientDiscriminator = arma::mat(gradient.memptr() + 
      gradientGenerator.n_elem,
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

  // If full batch seen then alternate training
  if (counter % numFunctions == 0)
  {
    // If we trained the discriminator for discIteration * 2
    if (!trainGenerator && iterationDiscriminator==1)
    {
      trainGenerator = true;
      CreateBatch();
      iterationDiscriminator = 2 * disIteration;
    }
    else
    {
      trainGenerator = false;
      offset = (offset + batchSize) % trainData.n_cols;
      CreateBatch();
      iterationDiscriminator--;
    }
  }
  size_t temp = gradientGenerator.n_rows;
  // Check the parameters
  CheckMatrices(parameters.rows(0, 5), generator.Parameters().rows(0, 5));
  CheckMatrices(parameters.rows(temp, temp +5), 
                discriminator.Parameters().rows(0, 5));

  // Check the Gradients
  /*
  std::cout << "Generator Gradients" << std::endl;
  gradient.rows(0, 5).print();
  std::cout << "Discriminator Gradient" << std::endl;
  gradient.rows(temp, temp + 5).print();
  */
  CheckMatrices(gradient.rows(0, 5), gradientGenerator.rows(0, 5));
  CheckMatrices(gradient.rows(temp, temp +5), 
                gradientDiscriminator.rows(0, 5));
  counter++;
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::Forward(arma::mat&& input)
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
    ganOutput = boost::apply_visitor(
        outputParameterVisitor,
        generator.network.back());
    discriminator.Forward(std::move(ganOutput));
}
}

template<typename Model, typename InitializationRuleType>
void GAN<Model, InitializationRuleType>::
Predict(arma::mat&& input, arma::mat& output)
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
} // namespace ann
} // namespace mlpack
# endif
