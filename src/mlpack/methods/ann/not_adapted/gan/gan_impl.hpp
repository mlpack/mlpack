/**
 * @file methods/ann/gan/gan_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_GAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_GAN_IMPL_HPP

#include "gan.hpp"

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/network_init.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
GAN<Model, InitializationRuleType, Noise, PolicyType>::GAN(
    Model generator,
    Model discriminator,
    InitializationRuleType& initializeRule,
    Noise& noiseFunction,
    const size_t noiseDim,
    const size_t batchSize,
    const size_t generatorUpdateStep,
    const size_t preTrainSize,
    const double multiplier,
    const double clippingParameter,
    const double lambda):
    generator(std::move(generator)),
    discriminator(std::move(discriminator)),
    initializeRule(initializeRule),
    noiseFunction(noiseFunction),
    noiseDim(noiseDim),
    numFunctions(0),
    batchSize(batchSize),
    currentBatch(0),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    multiplier(multiplier),
    clippingParameter(clippingParameter),
    lambda(lambda),
    reset(false),
    deterministic(false),
    genWeights(0),
    discWeights(0)
{
  // Insert IdentityLayer for joining the Generator and Discriminator.
  this->discriminator.network.insert(
      this->discriminator.network.begin(),
      new IdentityLayer<>());
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
GAN<Model, InitializationRuleType, Noise, PolicyType>::GAN(
    const GAN& network):
    predictors(network.predictors),
    responses(network.responses),
    generator(network.generator),
    discriminator(network.discriminator),
    initializeRule(network.initializeRule),
    noiseFunction(network.noiseFunction),
    noiseDim(network.noiseDim),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    multiplier(network.multiplier),
    clippingParameter(network.clippingParameter),
    lambda(network.lambda),
    reset(network.reset),
    currentBatch(network.currentBatch),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    noise(network.noise),
    deterministic(network.deterministic),
    genWeights(network.genWeights),
    discWeights(network.discWeights)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
GAN<Model, InitializationRuleType, Noise, PolicyType>::GAN(
    GAN&& network):
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    generator(std::move(network.generator)),
    discriminator(std::move(network.discriminator)),
    initializeRule(std::move(network.initializeRule)),
    noiseFunction(std::move(network.noiseFunction)),
    noiseDim(network.noiseDim),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    multiplier(network.multiplier),
    clippingParameter(network.clippingParameter),
    lambda(network.lambda),
    reset(network.reset),
    currentBatch(network.currentBatch),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    noise(std::move(network.noise)),
    deterministic(network.deterministic),
    genWeights(network.genWeights),
    discWeights(network.discWeights)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::ResetData(
    arma::mat trainData)
{
  currentBatch = 0;

  numFunctions = trainData.n_cols;
  noise.set_size(noiseDim, batchSize);

  deterministic = true;
  ResetDeterministic();

  /**
   * These predictors are shared by the discriminator network. The additional
   * batch size predictors are taken from the generator network while training.
   * For more details please look in EvaluateWithGradient() function.
   */
  this->predictors.set_size(trainData.n_rows, numFunctions + batchSize);
  this->predictors.cols(0, numFunctions - 1) = std::move(trainData);
  this->discriminator.predictors = arma::mat(this->predictors.memptr(),
      this->predictors.n_rows, this->predictors.n_cols, false, false);

  responses.ones(1, numFunctions + batchSize);
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      zeros(1, batchSize);
  this->discriminator.responses = arma::mat(this->responses.memptr(),
      this->responses.n_rows, this->responses.n_cols, false, false);

  this->generator.predictors.set_size(noiseDim, batchSize);
  this->generator.responses.set_size(predictors.n_rows, batchSize);

  if (!reset)
  {
    Reset();
  }
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::Reset()
{
  genWeights = 0;
  discWeights = 0;

  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);

  for (size_t i = 0; i < generator.network.size(); ++i)
  {
    genWeights += boost::apply_visitor(weightSizeVisitor, generator.network[i]);
  }

  for (size_t i = 0; i < discriminator.network.size(); ++i)
  {
    discWeights += boost::apply_visitor(weightSizeVisitor,
        discriminator.network[i]);
  }

  parameter.set_size(genWeights + discWeights, 1);
  generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
      false);
  discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
      discWeights, 1, false, false);

  // Initialize the parameters generator
  networkInit.Initialize(generator.network, parameter);
  // Initialize the parameters discriminator
  networkInit.Initialize(discriminator.network, parameter, genWeights);

  reset = true;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename OptimizerType, typename... CallbackTypes>
double GAN<Model, InitializationRuleType, Noise, PolicyType>::Train(
    arma::mat trainData,
    OptimizerType& Optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(trainData));

  return Optimizer.Optimize(*this, parameter, callbacks...);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Policy>
std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                 std::is_same_v<Policy, DCGAN>, double>
GAN<Model, InitializationRuleType, Noise, PolicyType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  currentInput = arma::mat(predictors.memptr() + (i * predictors.n_rows),
      predictors.n_rows, batchSize, false, false);
  currentTarget = arma::mat(responses.memptr() + i, 1, batchSize, false,
      false);

  discriminator.Forward(currentInput);
  double res = discriminator.outputLayer.Forward(
      boost::apply_visitor(
      outputParameterVisitor,
      discriminator.network.back()), currentTarget);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(noise);

  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  discriminator.Forward(predictors.cols(numFunctions,
      numFunctions + batchSize - 1));
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      zeros(1, batchSize);

  currentTarget = arma::mat(responses.memptr() + numFunctions,
      1, batchSize, false, false);
  res += discriminator.outputLayer.Forward(
      boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back()), currentTarget);

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename GradType, typename Policy>
std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                 std::is_same_v<Policy, DCGAN>, double>
GAN<Model, InitializationRuleType, Noise, PolicyType>::
EvaluateWithGradient(const arma::mat& /* parameters */,
                     const size_t i,
                     GradType& gradient,
                     const size_t /* batchSize */)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (gradient.is_empty())
  {
    if (parameter.is_empty())
      Reset();
    gradient = zeros<arma::mat>(parameter.n_elem, 1);
  }
  else
    gradient.zeros();

  if (this->deterministic)
  {
    this->deterministic = false;
    ResetDeterministic();
  }

  if (noiseGradientDiscriminator.is_empty())
  {
    noiseGradientDiscriminator = zeros<arma::mat>(
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

  // Get the gradients of the Discriminator.
  double res = discriminator.EvaluateWithGradient(discriminator.parameter,
      i, gradientDiscriminator, batchSize);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(noise);
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      zeros(1, batchSize);

  // Get the gradients of the Generator.
  res += discriminator.EvaluateWithGradient(discriminator.parameter,
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(noise))).
    // Pass the error from Discriminator to Generator.
    responses.cols(numFunctions, numFunctions + batchSize - 1) =
        ones(1, batchSize);

    discriminator.outputLayer.Backward(
        boost::apply_visitor(outputParameterVisitor,
        discriminator.network.back()), discriminator.responses.cols(
        numFunctions, numFunctions + batchSize - 1), discriminator.error);
    discriminator.Backward();

    generator.error = boost::apply_visitor(deltaVisitor,
        discriminator.network[1]);

    generator.Predictors() = noise;
    generator.Backward();
    generator.ResetGradients(gradientGenerator);
    generator.Gradient(generator.Predictors().cols(0, batchSize - 1));

    gradientGenerator *= multiplier;
  }

  currentBatch++;


  if (preTrainSize > 0)
  {
    preTrainSize--;
  }

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Policy>
std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                 std::is_same_v<Policy, DCGAN>, void>
GAN<Model, InitializationRuleType, Noise, PolicyType>::
Gradient(const arma::mat& parameters,
         const size_t i,
         arma::mat& gradient,
         const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::Shuffle()
{
  const arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      numFunctions - 1, numFunctions));
  predictors.cols(0, numFunctions - 1) = predictors.cols(ordering);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::Forward(
    const arma::mat& input)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  generator.Forward(input);
  arma::mat ganOutput = boost::apply_visitor(outputParameterVisitor,
      generator.network.back());

  discriminator.Forward(ganOutput);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::
Predict(arma::mat input, arma::mat& output)
{
  if (parameter.is_empty())
  {
    Reset();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  Forward(input);

  output = boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back());
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::
ResetDeterministic()
{
  this->discriminator.deterministic = deterministic;
  this->generator.deterministic = deterministic;
  this->discriminator.ResetDeterministic();
  this->generator.ResetDeterministic();
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Archive>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::
serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(parameter));
  ar(CEREAL_NVP(generator));
  ar(CEREAL_NVP(discriminator));
  ar(CEREAL_NVP(reset));
  ar(CEREAL_NVP(genWeights));
  ar(CEREAL_NVP(discWeights));

  if (cereal::is_loading<Archive>())
  {
    // Share the parameters between the network.
    generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
        false);
    discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
        discWeights, 1, false, false);

    size_t offset = 0;
    for (size_t i = 0; i < generator.network.size(); ++i)
    {
      offset += boost::apply_visitor(WeightSetVisitor(
          generator.parameter, offset), generator.network[i]);

      boost::apply_visitor(resetVisitor, generator.network[i]);
    }

    offset = 0;
    for (size_t i = 0; i < discriminator.network.size(); ++i)
    {
      offset += boost::apply_visitor(WeightSetVisitor(
          discriminator.parameter, offset), discriminator.network[i]);

      boost::apply_visitor(resetVisitor, discriminator.network[i]);
    }

    deterministic = true;
    ResetDeterministic();
  }
}

} // namespace mlpack
# endif
