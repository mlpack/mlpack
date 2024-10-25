/**
 * @file methods/ann/gan/wgangp_impl.hpp
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_WGANGP_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_WGANGP_IMPL_HPP

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
template<typename Policy>
std::enable_if_t<std::is_same_v<Policy, WGANGP>, double>
GAN<Model, InitializationRuleType, Noise, PolicyType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  if ((parameter.is_empty()))
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

  discriminator.Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
      outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget));

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));

  arma::mat generatedData = boost::apply_visitor(outputParameterVisitor,
      generator.network.back());
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      generatedData;
  discriminator.Forward(std::move(predictors.cols(numFunctions,
      numFunctions + batchSize - 1)));
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -ones(1, batchSize);

  currentTarget = arma::mat(responses.memptr() + numFunctions,
      1, batchSize, false, false);
  res += discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
      outputParameterVisitor,
      discriminator.network.back())), std::move(currentTarget));

  // Gradient Penalty is calculated here.
  double epsilon = Random();
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      (epsilon * currentInput) + ((1.0 - epsilon) * generatedData);
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -ones(1, batchSize);
  discriminator.Gradient(discriminator.parameter, numFunctions,
      normGradientDiscriminator, batchSize);
  res += lambda * std::pow(norm(normGradientDiscriminator, 2) - 1, 2);

  return res;
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
template<typename GradType, typename Policy>
std::enable_if_t<std::is_same_v<Policy, WGANGP>, double>
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

  currentInput = arma::mat(predictors.memptr() + (i * predictors.n_rows),
      predictors.n_rows, batchSize, false, false);

  // Get the gradients of the Discriminator.
  double res = discriminator.EvaluateWithGradient(discriminator.parameter,
      i, gradientDiscriminator, batchSize);

  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  arma::mat generatedData = boost::apply_visitor(outputParameterVisitor,
      generator.network.back());

  // Gradient Penalty is calculated here.
  double epsilon = Random();
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      (epsilon * currentInput) + ((1.0 - epsilon) * generatedData);
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -ones(1, batchSize);
  discriminator.Gradient(discriminator.parameter, numFunctions,
      normGradientDiscriminator, batchSize);
  res += lambda * std::pow(norm(normGradientDiscriminator, 2) - 1, 2);

  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      generatedData;
  res += discriminator.EvaluateWithGradient(discriminator.parameter,
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -D(G(noise)).
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
std::enable_if_t<std::is_same_v<Policy, WGANGP>, void>
GAN<Model, InitializationRuleType, Noise, PolicyType>::
Gradient(const arma::mat& parameters,
         const size_t i,
         arma::mat& gradient,
         const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

} // namespace mlpack
# endif
