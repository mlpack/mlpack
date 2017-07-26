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
#include <chrono>
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
    size_t disIteration,
    size_t generatorInSize):
    trainData(trainData),
    initializeRule(initializeRule),
    generator(generator),
    discriminator(discriminator),
    trainGenerator(false),
    batchSize(batchSize),
    iterations(iterations),
    disIteration(disIteration),
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

  // Todo remove this
  size_t idx = RandInt(0, parameter.n_rows);
  std::cout << "idx" << idx << std::endl;
  assert(discriminator.Parameters()(idx) == parameter(generator.Parameters().n_rows + idx));
  assert(generator.Parameters()(idx) == parameter(idx));
  // 
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
  // Todo remove this
  assert(boost::apply_visitor(weightSizeVisitor, discriminator.network[idx]) > 0);
  assert(boost::apply_visitor(weightSizeVisitor, generator.network[idx]) > 0);
  //
  reset = true;
}

template<typename Generator, typename Discriminator, typename IntializerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::GenerateData(arma::mat& data, arma::mat& labels, size_t offset)
{
  if (disIteration * batchSize > trainData.n_cols)
  {
    std::cout << "k value too large" << std::endl;
    return;
  }
  // enery iteration since std::move
  data.set_size(trainData.n_rows, (batchSize * disIteration * 2) + batchSize);
  labels.set_size(1, (batchSize * disIteration * 2) + batchSize);
  noiseData.set_size(generatorInSize, batchSize * disIteration + batchSize);

  size_t endColRealData = 2 * batchSize * disIteration;
  size_t endColFakeData = batchSize * disIteration;
  size_t n_rows = trainData.n_rows;

  if (!trainGenerator)
  {
    // fake data discriminator
    disFakeData = arma::mat(data.memptr(), n_rows,
        batchSize * disIteration, false, false);
    // real data
    data.cols(endColFakeData, endColRealData - 1) =
        arma::mat(trainData.memptr() + offset, n_rows, batchSize * disIteration,
        true, false);
    
    Generate(std::move(disFakeData), std::move(noiseData.cols(0,
        endColFakeData - 1)));

    labels.cols(0, endColFakeData - 1).zeros();
    labels.cols(endColFakeData, endColRealData - 1).ones();

    this->predictors = std::move(data.cols(0, endColRealData - 1));
    this->responses = std::move(labels.cols(0, endColRealData - 1));
    assert(trainData.n_rows == n_rows);
    // Todo Remove this
    size_t idx = RandInt(endColFakeData, endColRealData - 1);

    numFunctions = predictors.n_cols;
    discriminator.predictors = this->predictors;
    discriminator.responses = this->responses;
    generator.predictors = std::move(noiseData.cols(0, endColFakeData - 1));
    generator.responses = this->responses;
  }
  else
  {
    // fake data generator
    genFakeData = arma::mat(data.memptr() + disFakeData.n_elem, n_rows,
        batchSize, false, false);
    Generate(std::move(genFakeData), std::move(noiseData.cols(endColFakeData,
        noiseData.n_cols - 1)));
    // real label for generator's fake data
    labels.cols(endColRealData, labels.n_cols - 1).ones();
    this->predictors = std::move(noiseData.cols(endColFakeData,
        noiseData.n_cols - 1));
    this->responses = std::move(labels.cols(endColRealData, labels.n_cols - 1));
    assert(trainData.n_rows == n_rows);

    numFunctions = predictors.n_cols;
    discriminator.predictors = std::move(genFakeData);
    discriminator.responses = this->responses;
    generator.predictors = this->predictors;
    generator.responses = this->responses;
  }
}

template<typename Generator, typename Discriminator, typename IntializerType>
template<typename OptimizerType>
void GenerativeAdversarialNetwork<Generator, Discriminator, IntializerType>
::Train(OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  std::chrono::time_point<std::chrono::system_clock> start, end;
  for (size_t i = 0; i < iterations; i++)
  {
    std::cout << "iteration #" << i << std::endl;
    start = std::chrono::system_clock::now();
    size_t offset = 0;
    // last batch
    if (offset + batchSize >  trainData.n_cols)
      offset -=  offset + batchSize - trainData.n_cols;

    // Generate fake data for discrminator to train on.
    GenerateData(data, labels, offset);
    offset += batchSize * disIteration;
    Optimizer.Optimize(*this, parameter);

    // Training Generator
    trainGenerator = true;
    GenerateData(data, labels, 0);
    Optimizer.Optimize(*this, parameter);
    // set train generator to false
    trainGenerator = false;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout<< "elapsed time: " << elapsed_seconds.count() << "s\n";
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
