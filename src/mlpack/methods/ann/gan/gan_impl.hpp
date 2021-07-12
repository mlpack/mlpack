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
namespace ann /** Artifical Neural Network.  */ {
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::GAN(
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::GAN(
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::GAN(
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::ResetData(
    InputType trainData)
{
  currentBatch = 0;

  numFunctions = trainData.n_cols;
  noise.set_size(noiseDim, batchSize);

  deterministic = false;
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
      arma::zeros(1, batchSize);
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Reset()
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename OptimizerType, typename... CallbackTypes>
double GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Train(
    InputType predictors,
    OptimizerType& Optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors));

  return Optimizer.Optimize(*this, parameter, callbacks...);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
  >
template<typename Policy,
         typename DiscOptimizerType,
         typename GenOptimizerType,
         typename... CallbackTypes>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Train(
    InputType predictors,
    DiscOptimizerType& discriminatorOptimizer,
    GenOptimizerType& generatorOptimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors));
  const size_t maxIterations = discriminatorOptimizer.MaxIterations();

  // We pass two batches during training hence maxIterations is doubled.
  discriminatorOptimizer.MaxIterations() =
      discriminatorOptimizer.MaxIterations() * 2;

  //TODO : find a way to sync shuffle between disc and gen
  // Avoid shuffling of predictors during training.
  discriminatorOptimizer.Shuffle() = false;
  generatorOptimizer.Shuffle() = false;

  // Predictors and responses for generator and discriminator network.
  InputType discriminatorPredictors;
  OutputType discriminatorResponses;

  double ObjValue = 0;

  //const size_t actualMaxIterations = (maxIterations == 0) ?
    //  std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < maxIterations; i++)
  {
    // Is this iteration the start of a sequence?
    if (currentBatch % numFunctions == 0 && i > 0)
    {
      currentBatch = 0;
    }

    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of functions left.
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions -
        currentBatch);

    // Training data for dicriminator.
    if (effectiveBatchSize != batchSize)
    {
      noise.set_size(noiseDim, effectiveBatchSize);
      discriminatorOptimizer.BatchSize() = effectiveBatchSize;
      discriminatorOptimizer.MaxIterations() = effectiveBatchSize * 2;
    }
    noise.imbue( [&]() { return noiseFunction();} );
    InputType fakeImages;
    generator.Forward(noise, fakeImages);

    discriminatorPredictors = arma::join_rows(
         predictors.cols(currentBatch,  currentBatch + effectiveBatchSize -
        1), fakeImages);

    discriminatorResponses = arma::join_rows(arma::ones(1, effectiveBatchSize),
        arma::zeros(1, effectiveBatchSize));

    // Train the discriminator.
    ObjValue+= discriminator.Train(discriminatorPredictors, discriminatorResponses,
        discriminatorOptimizer, callbacks...);

    if (effectiveBatchSize != batchSize)
    {
      noise.set_size(noiseDim, batchSize);
      discriminatorOptimizer.BatchSize() = batchSize;
      discriminatorOptimizer.MaxIterations() = batchSize * 2;
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
          discriminator.error);
      discriminator.Backward();

      generator.error = boost::apply_visitor(deltaVisitor,
          discriminator.network[1]);

      // Train the generator network.
      ObjValue+= generator.Train(noise, generatorOptimizer, callbacks...);
    }

    if (preTrainSize > 0)
    {
      preTrainSize--;
    }

    currentBatch += effectiveBatchSize;
  }

  // Changing maxIterations back to normal.
  discriminatorOptimizer.MaxIterations() =
      discriminatorOptimizer.MaxIterations() / 2;
  return ObjValue;
}


template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Evaluate(
    const InputType& /* parameters */,
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
      arma::zeros(1, batchSize);

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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename GradType, typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, double>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
EvaluateWithGradient(const InputType& /* parameters */,
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
    gradient = arma::zeros<arma::mat>(parameter.n_elem, 1);
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
  generator.Forward(noise);
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename Policy>
typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                        std::is_same<Policy, DCGAN>::value, void>::type
GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
Gradient(const InputType& parameters,
         const size_t i,
         OutputType& gradient,
         const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Shuffle()
{
  const arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      numFunctions - 1, numFunctions));
  predictors.cols(0, numFunctions - 1) = predictors.cols(ordering);
}

template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::Forward(
    const InputType& input)
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
Predict(InputType& input, OutputType& output)
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
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
  typename PolicyType,
  typename InputType,
  typename OutputType
>
template<typename Archive>
void GAN<Model, InitializationRuleType, Noise, PolicyType, InputType, OutputType>::
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

} // namespace ann
} // namespace mlpack
# endif
