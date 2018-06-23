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
#ifndef MLPACK_METHODS_ANN_GAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_IMPL_HPP

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
    Model& generator,
    Model& discriminator,
    InitializationRuleType initializeRule,
    Noise noiseFunction,
    size_t noiseDim,
    size_t batchSize,
    size_t generatorUpdateStep,
    size_t preTrainSize,
    double multiplier,
    double clippingParameter,
    double lambda):
    predictors(predictors),
    generator(generator),
    discriminator(discriminator),
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
  discriminator.network.insert(
      discriminator.network.begin(),
      new IdentityLayer<>());

  counter = 0;
  currentBatch = 0;

  discriminator.deterministic = generator.deterministic = true;

  responses.set_size(1, predictors.n_cols);
  responses.ones();

  discriminator.predictors.set_size(predictors.n_rows,
      predictors.n_cols + batchSize);
  discriminator.predictors.cols(0, predictors.n_cols - 1) = predictors;

  discriminator.responses.set_size(1, predictors.n_cols + batchSize);
  discriminator.responses.ones();
  discriminator.responses.cols(predictors.n_cols,
      predictors.n_cols + batchSize - 1) = arma::zeros(1, batchSize);

  numFunctions = predictors.n_cols;

  noise.set_size(noiseDim, batchSize);

  generator.predictors.set_size(noiseDim, batchSize);
  generator.responses.set_size(predictors.n_rows, batchSize);
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
void GAN<Model, InitializationRuleType, Noise, PolicyType>::Train(
    OptimizerType& Optimizer)
{
  if (!reset)
    Reset();
  Optimizer.Optimize(*this, parameter);
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

  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  discriminator.Forward(std::move(discriminator.predictors.cols(numFunctions,
      numFunctions + batchSize - 1)));
  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);

  currentTarget = arma::mat(discriminator.responses.memptr() + numFunctions,
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
template<typename Policy>
typename std::enable_if<std::is_same<Policy, WGAN>::value, double>::type
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

  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  discriminator.Forward(std::move(discriminator.predictors.cols(numFunctions,
      numFunctions + batchSize - 1)));
  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);

  currentTarget = arma::mat(discriminator.responses.memptr() + numFunctions,
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
template<typename Policy>
typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                        double>::type
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

  arma::mat generatedData = boost::apply_visitor(outputParameterVisitor,
      generator.network.back());
  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      generatedData;
  discriminator.Forward(std::move(discriminator.predictors.cols(numFunctions,
      numFunctions + batchSize - 1)));
  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);

  currentTarget = arma::mat(discriminator.responses.memptr() + numFunctions,
      1, batchSize, false, false);
  res += discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
      outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget));

  // Gradient Penalty is calculated here.
  double epsilon = math::Random();
  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      (epsilon * currentInput) + ((1.0 - epsilon) * generatedData);
  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);
  discriminator.Gradient(discriminator.parameter, numFunctions,
      normGradientDiscriminator, batchSize);
  res += lambda * std::pow(arma::norm(normGradientDiscriminator, 2) - 1, 2);

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
Gradient(const arma::mat& /* parameters */,
         const size_t i,
         arma::mat& gradient,
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
  discriminator.Gradient(discriminator.parameter, i, gradientDiscriminator,
      batchSize);
  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());

  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      arma::zeros(1, batchSize);
  discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(noise))).
    // Pass the error from Discriminator to Generator.
    discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
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
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Policy>
typename std::enable_if<std::is_same<Policy, WGAN>::value, void>::type
GAN<Model, InitializationRuleType, Noise, PolicyType>::
Gradient(const arma::mat& /* parameters */,
         const size_t i,
         arma::mat& gradient,
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
  discriminator.Gradient(discriminator.parameter, i, gradientDiscriminator,
      batchSize);
  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());

  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);
  discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;
  gradientDiscriminator = arma::clamp(gradientDiscriminator,
      -clippingParameter, clippingParameter);

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -D(G(noise)).
    // Pass the error from Discriminator to Generator.
    discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
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
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename Policy>
typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                        void>::type
GAN<Model, InitializationRuleType, Noise, PolicyType>::
Gradient(const arma::mat& /* parameters */,
         const size_t i,
         arma::mat& gradient,
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
  discriminator.Gradient(discriminator.parameter, i, gradientDiscriminator,
      batchSize);
  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  discriminator.predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());

  discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);
  discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -D(G(noise)).
    // Pass the error from Discriminator to Generator.
    discriminator.responses.cols(numFunctions, numFunctions + batchSize - 1) =
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
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
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
