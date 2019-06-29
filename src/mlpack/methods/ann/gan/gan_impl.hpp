/**
 * @file gan_impl.hpp
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
#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network.  */ {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
GAN<Model, InitializationRuleType, Noise, PolicyType>::GAN(
    arma::mat& predictors,
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
    batchSize(batchSize),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    multiplier(multiplier),
    clippingParameter(clippingParameter),
    lambda(lambda),
    reset(false)
{
  // Insert IdentityLayer for joining the Generator and Discriminator.
  this->discriminator.network.insert(
      this->discriminator.network.begin(),
      new IdentityLayer<>());

  counter = 0;
  currentBatch = 0;

  this->discriminator.deterministic = this->generator.deterministic = true;

  this->predictors.set_size(predictors.n_rows, predictors.n_cols + batchSize);
  this->predictors.cols(0, predictors.n_cols - 1) = predictors;
  this->discriminator.predictors = arma::mat(this->predictors.memptr(),
      this->predictors.n_rows, this->predictors.n_cols, false, false);

  responses.ones(1, predictors.n_cols + batchSize);
  responses.cols(predictors.n_cols,
      predictors.n_cols + batchSize - 1) = arma::zeros(1, batchSize);
  this->discriminator.responses = arma::mat(this->responses.memptr(),
      this->responses.n_rows, this->responses.n_cols, false, false);

  numFunctions = predictors.n_cols;

  noise.set_size(noiseDim, batchSize);

  this->generator.predictors.set_size(noiseDim, batchSize);
  this->generator.responses.set_size(predictors.n_rows, batchSize);
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
    counter(network.counter),
    currentBatch(network.currentBatch),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    noise(network.noise)
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
    counter(network.counter),
    currentBatch(network.currentBatch),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    noise(std::move(network.noise))
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::Reset()
{
  size_t genWeights = 0;
  size_t discWeights = 0;

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
template<typename OptimizerType>
double GAN<Model, InitializationRuleType, Noise, PolicyType>::Train(
    OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  return Optimizer.Optimize(*this, parameter);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  if (!reset)
    Reset();

  currentInput = arma::mat(predictors.memptr() + (i * predictors.n_rows),
      predictors.n_rows, batchSize, false, false);
  currentTarget = arma::mat(responses.memptr() + i, 1, batchSize, false,
      false);

  discriminator.Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
      outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget));

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));

  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  discriminator.Forward(std::move(predictors.cols(numFunctions,
      numFunctions + batchSize - 1)));
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);

  currentTarget = arma::mat(responses.memptr() + numFunctions,
      1, batchSize, false, false);
  res += discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
      outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget));

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename GradType, typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType>::
EvaluateWithGradient(const arma::mat& /* parameters */,
                     const size_t i,
                     GradType& gradient,
                     const size_t /* batchSize */)
{
  if (!reset)
    Reset();

  if (gradient.is_empty())
  {
    if (parameter.is_empty())
      Reset();
    gradient = arma::zeros<arma::mat>(parameter.n_elem, 1);
  }
  else
    gradient.zeros();

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

  // Get the gradients of the Discriminator.
  double res = discriminator.EvaluateWithGradient(discriminator.parameter,
      i, gradientDiscriminator, batchSize);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);

  // Get the gradients of the Generator.
  res += discriminator.EvaluateWithGradient(discriminator.parameter,
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(noise))).
    // Pass the error from Discriminator to Generator.
    responses.cols(numFunctions, numFunctions + batchSize - 1) =
        arma::ones(1, batchSize);
    discriminator.Gradient(discriminator.parameter, numFunctions,
        noiseGradientDiscriminator, batchSize);
    generator.error = boost::apply_visitor(deltaVisitor,
        discriminator.network[1]);

    generator.Predictors() = noise;
    generator.ResetGradients(gradientGenerator);
    generator.Gradient(generator.parameter, 0, gradientGenerator, batchSize);

    gradientGenerator *= multiplier;
  }

  counter++;
  currentBatch++;

  // Revert the counter to zero, if the total dataset get's covered.
  if (counter * batchSize >= numFunctions)
  {
    counter = 0;
  }

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
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, void>::type
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
    arma::mat&& input)
{
  if (!reset)
    Reset();

  generator.Forward(std::move(input));
  ganOutput = boost::apply_visitor(
      outputParameterVisitor,
      generator.network.back());

  discriminator.Forward(std::move(ganOutput));
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::
Predict(arma::mat&& input, arma::mat& output)
{
  if (!reset)
    Reset();

  Forward(std::move(input));

  output = boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back());
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Archive>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::
serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(generator);
  ar & BOOST_SERIALIZATION_NVP(discriminator);
  ar & BOOST_SERIALIZATION_NVP(noiseFunction);
}

} // namespace ann
} // namespace mlpack
# endif
