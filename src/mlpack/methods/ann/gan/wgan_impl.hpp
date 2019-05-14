/**
 * @file wgan_impl.hpp
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_WGAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_WGAN_IMPL_HPP

#include "gan.hpp"

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/network_init.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network.  */ {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType
>
<<<<<<< HEAD
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

  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  discriminator.Forward(std::move(predictors.cols(numFunctions,
      numFunctions + batchSize - 1)));
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);

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
typename std::enable_if<std::is_same<Policy, WGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType>::
EvaluateWithGradient(const arma::mat& /* parameters */,
                     const size_t i,
                     GradType& gradient,
                     const size_t /* batchSize */)
=======
template<typename Policy, typename DiscOptimizerType, typename GenOptimizerType>
typename std::enable_if<std::is_same<Policy, WGAN>::value, void>::type
GAN<Model, InitializationRuleType, Noise, PolicyType>::Train(
    arma :: mat trainData,
    DiscOptimizerType& discriminatorOptimizer,
    GenOptimizerType& generatorOptimizer,
    size_t maxIterations,
    size_t discIterations)
>>>>>>> Implement WGAN with dual optimizer.
{
  numFunctions = trainData.n_cols;
  noise.set_size(noiseDim, batchSize);

  // To keep track of where we are.
  size_t currentFunction = 0;

  // We pass two batches during training hence maxIterations is doubled.
  discriminatorOptimizer.MaxIterations() =
      discriminatorOptimizer.MaxIterations() * 2;

<<<<<<< HEAD
  noise.imbue( [&]() { return noiseFunction();} );
  generator.Forward(std::move(noise));
  predictors.cols(numFunctions, numFunctions + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generator.network.back());
  responses.cols(numFunctions, numFunctions + batchSize - 1) =
      -arma::ones(1, batchSize);

  // Get the gradients of the Generator.
  res += discriminator.EvaluateWithGradient(discriminator.parameter,
      numFunctions, noiseGradientDiscriminator, batchSize);
  gradientDiscriminator += noiseGradientDiscriminator;
  gradientDiscriminator = arma::clamp(gradientDiscriminator,
      -clippingParameter, clippingParameter);

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -D(G(noise)).
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
=======
  // Predictors and responses for generator and discriminator network.
  arma::mat discriminatorPredictors;
  arma::mat discriminatorResponses;

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; i++)
>>>>>>> Implement WGAN with dual optimizer.
  {
    for (size_t j = 0; j < discIterations; j++)
    {
      // Is this iteration the start of a sequence?
      if (currentFunction % numFunctions == 0 && (i > 0 || j > 0))
      {
        currentFunction = 0;
      }

      // Find the effective batch size; we have to take the minimum of three
      // things:
      // - the batch size can't be larger than the user-specified batch size;
      // - the batch size can't be larger than the number of functions left.
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions -
          currentFunction);

      // Training data for dicriminator.
      if (effectiveBatchSize != batchSize)
      {
        noise.set_size(noiseDim, effectiveBatchSize);
        discriminatorOptimizer.BatchSize() = effectiveBatchSize;
        discriminatorOptimizer.MaxIterations() = effectiveBatchSize * 2;
      }
      noise.imbue( [&]() { return noiseFunction();} );
      arma::mat fakeImages;
      generator.Forward(noise, fakeImages);

      discriminatorPredictors = arma::join_rows(
          trainData.cols(currentFunction,  currentFunction + effectiveBatchSize
          - 1), fakeImages);

      discriminatorResponses = arma::join_rows(arma::ones(1,
          effectiveBatchSize), arma::zeros(1, effectiveBatchSize));

      // Train the discriminator.
      discriminator.Train(discriminatorPredictors, discriminatorResponses,
          discriminatorOptimizer);

      // Clip the weights of discriminator network.
      discriminator.Parameters() = arma::clamp(discriminator.Parameters(),
          -clippingParameter, clippingParameter);


      if (effectiveBatchSize != batchSize)
      {
        noise.set_size(noiseDim, batchSize);
        discriminatorOptimizer.BatchSize() = batchSize;
        discriminatorOptimizer.MaxIterations() = batchSize * 2;
      }

      currentFunction += effectiveBatchSize;
    }
    if (preTrainSize == 0)
    {
      // Calculate error for generator network.
      discriminatorResponses = arma::ones(1, batchSize);

      noise.imbue( [&]() { return noiseFunction();} );
      generator.Forward(std::move(noise));

      discriminator.Forward(std::move(boost::apply_visitor(
          outputParameterVisitor, generator.network.back())));

      discriminator.outputLayer.Backward(
          std::move(boost::apply_visitor(outputParameterVisitor,
          discriminator.network.back())), std::move(discriminatorResponses),
          std::move(discriminator.error));
      discriminator.Backward();

      generator.error = boost::apply_visitor(deltaVisitor,
          discriminator.network[1]);

      // Train the generator network.
      generator.Train(noise, generatorOptimizer);
    }

    if (preTrainSize > 0)
    {
      preTrainSize--;
    }
  }

  // Changing maxIterations back to normal.
  discriminatorOptimizer.MaxIterations() =
      discriminatorOptimizer.MaxIterations() / 2;
}

} // namespace ann
} // namespace mlpack
# endif
