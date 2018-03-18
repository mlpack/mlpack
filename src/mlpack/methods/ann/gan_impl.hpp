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
    size_t preTrainSize,
    double multiplier):
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
    reset(false)
{
  std::cout << "Enter Constructor:" << std::endl;
  std::cout << "Inserting Identity Layer for joining Generator and Discriminator" << std::endl;
  // Insert IdentityLayer for joining the Generator and Discriminator.
  discriminator.network.insert(
      discriminator.network.begin(),
      new IdentityLayer<>());

  counter = 0;
  currentBatch = 0;
  std::cout << "Counter: " << counter << std::endl;
  std::cout << "CurrentBatch: " << currentBatch << std::endl;

  discriminator.deterministic = generator.deterministic = true;
  std::cout << "Deterministic: " << discriminator.deterministic << std::endl;

  responses.set_size(1, predictors.n_cols);
  responses.ones();
  //std::cout << "Responses:" << std::endl << responses << std::endl;

  //std::cout << "Predictors:" << std::endl << predictors << std::endl;
  discriminator.predictors.set_size(predictors.n_rows, predictors.n_cols + 1);
  discriminator.predictors.cols(0, predictors.n_cols - 1) = predictors;
  //std::cout << "Discriminator Predictors:" << std::endl << discriminator.predictors << std::endl;

  discriminator.responses.set_size(1, predictors.n_cols + 1);
  discriminator.responses.ones();
  discriminator.responses(predictors.n_cols) = 0;
  //std::cout << "Discriminator Responses:" << std::endl << discriminator.responses << std::endl;

  numFunctions = predictors.n_cols;
  std::cout << "Number of Functions: " << numFunctions << std::endl;

  noise.set_size(noiseDim, 1);

  generator.predictors.set_size(noiseDim, noise.n_cols + 1);
  generator.responses.set_size(predictors.n_rows, noise.n_cols + 1);
  //std::cout << "Noise:" << std::endl << noise << std::endl;
  std::cout << "Exit Constructor:" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::Reset()
{
  std::cout << "Enter Reset:" << std::endl;
  size_t genWeights = 0;
  size_t discWeights = 0;
  //std::cout << "Generator Weights: " << genWeights << std::endl;
  //std::cout << "Discriminator Weights: " << discWeights << std::endl;
  std::cout << "Calling Network Initialization:" << std::endl;
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);

  std::cout << "Updating Generator Weights:" << std::endl;
  for (size_t i = 0; i < generator.network.size(); ++i){
    std::cout << "Iteration " << i << ": " << boost::apply_visitor(weightSizeVisitor, generator.network[i]) << std::endl;
    genWeights += boost::apply_visitor(weightSizeVisitor, generator.network[i]);
    std::cout << "Updated Generator Weights: " << genWeights << std::endl;
  }

  std::cout << "Updating Discriminator Weights:" << std::endl;
  for (size_t i = 0; i < discriminator.network.size(); ++i){
    std::cout << "Iteration " << i << ": " << boost::apply_visitor(weightSizeVisitor, discriminator.network[i]) << std::endl;
    discWeights += boost::apply_visitor(weightSizeVisitor,
        discriminator.network[i]);
    std::cout << "Updated Discriminator Weights: " << discWeights << std::endl;
  }

  parameter.set_size(genWeights + discWeights, 1);
  //std::cout << "Parameters:" << std::endl << parameter << std::endl;
  generator.Parameters() = arma::mat(parameter.memptr(), genWeights, 1, false,
      false);
  //std::cout << "Generator Parameters:" << std::endl << generator.Parameters() << std::endl;
  discriminator.Parameters() = arma::mat(parameter.memptr() + genWeights,
      discWeights, 1 , false, false);
  //std::cout << "Discriminator Parameters:" << std::endl << discriminator.Parameters() << std::endl;

  std::cout << "Initializing Network with Generator:" << std::endl;
  // Initialize the parameters generator
  networkInit.Initialize(generator.network, parameter);
  std::cout << "Initializing Network with Discriminator:" << std::endl;
  // Initialize the parameters discriminator
  networkInit.Initialize(discriminator.network, parameter, genWeights);

  reset = true;
  std::cout << "Reset: " << reset << std::endl;
  std::cout << "Exit Reset:" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
template<typename OptimizerType>
void GAN<Model, InitializationRuleType, Noise>::Train(
    OptimizerType& Optimizer)
{
  std::cout << "Enter GAN Training:" << std::endl;
  if (!reset)
    Reset();
  std::cout << "Calling Optimizer:" << std::endl;
  Optimizer.Optimize(*this, parameter);
  std::cout << "Exit GAN Training:" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
double GAN<Model, InitializationRuleType, Noise>::Evaluate(
    const arma::mat& /*parameters*/,
    const size_t i,
    const size_t /*batchSize*/)
{
  std::cout << "Enter GAN Evaluate:" << std::endl;
  std::cout << "Iteration: " << i << std::endl;
  if (!reset)
    Reset();

  currentInput = this->predictors.unsafe_col(i);
  //std::cout << "Current Input:" << std::endl << currentInput << std::endl;
  currentTarget = this->responses.unsafe_col(i);
  //std::cout << "Current Target:" << std::endl << currentTarget << std::endl;
  std::cout << "Calling Discriminator Forward:" << std::endl;
  discriminator.Forward(std::move(currentInput));
  double res = discriminator.outputLayer.Forward(
      std::move(boost::apply_visitor(
          outputParameterVisitor,
          discriminator.network.back())), std::move(currentTarget));
  std::cout << "Result: " << res << std::endl;
  noise.imbue( [&]() { return noiseFunction();} );
  //std::cout << "Noise:" << std::endl << noise << std::endl;
  std::cout << "Calling Generator Forward:" << std::endl;
  generator.Forward(std::move(noise));
  arma::mat temp = boost::apply_visitor(
      outputParameterVisitor, generator.network.back());
  //std::cout << "Temp:" << std::endl << temp << std::endl;

  std::cout << "Number of Functions: " << numFunctions << std::endl;
  //std::cout << "Discriminator Predictors:" << std::endl << discriminator.predictors << std::endl;
  discriminator.predictors.col(numFunctions) = temp;
  //std::cout << "Updated Discriminator Predictors:" << std::endl << discriminator.predictors << std::endl;
  std::cout << "Calling Discriminator Forward:" << std::endl;
  discriminator.Forward(std::move(discriminator.predictors.col(numFunctions)));
  //std::cout << "Discriminator Responses:" << std::endl << discriminator.responses << std::endl;
  discriminator.responses(numFunctions) = 0;
  //std::cout << "Updated Discriminator Responses:" << std::endl << discriminator.responses << std::endl;

  currentTarget = discriminator.responses.unsafe_col(numFunctions);
  //std::cout << "Updated Current Target:" << std::endl << currentTarget << std::endl;
  res += discriminator.outputLayer.Forward(
        std::move(boost::apply_visitor(
            outputParameterVisitor,
            discriminator.network.back())), std::move(currentTarget));
  std::cout << "Updated Result: " << res << std::endl;
  std::cout << "Exit GAN Evaluate:" << std::endl;
  return res;
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::
Gradient(const arma::mat& /*parameters*/, const size_t i, arma::mat& gradient,
    const size_t /*batchSize*/)
{
  std::cout << "Enter GAN Gradient:" << std::endl;
  if (!reset){
    std::cout << "Calling Reset(1)" << std::endl;
    Reset();
  }

  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      std::cout << "Calling Reset(2)" << std::endl;
      Reset();
    }
    gradient = arma::zeros<arma::mat>(parameter.n_elem, 1);
    //std::cout << "Gradient(1):" << std::endl << gradient << std::endl;
  }
  else
  {
    gradient.zeros();
    //std::cout << "Gradient(2):" << std::endl << gradient << std::endl;
  }

  if (noiseGradientDiscriminator.is_empty())
  {
    noiseGradientDiscriminator = arma::zeros<arma::mat>(
        gradientDiscriminator.n_elem, 1);
    //std::cout << "Noise Gradient Discriminator(1):" << std::endl << noiseGradientDiscriminator << std::endl;
  }
  else
  {
    noiseGradientDiscriminator.zeros();
    //std::cout << "Noise Gradient Discriminator(2):" << std::endl << noiseGradientDiscriminator << std::endl;
  }

  gradientGenerator = arma::mat(gradient.memptr(),
      generator.Parameters().n_elem, 1, false, false);
  //std::cout << "Generator Gradient:" << std::endl << gradientGenerator << std::endl;

  gradientDiscriminator = arma::mat(gradient.memptr() +
      gradientGenerator.n_elem,
      discriminator.Parameters().n_elem, 1, false, false);
  //std::cout << "Discriminator Gradient:" << std::endl << gradientDiscriminator << std::endl;

  // Get the gradients of the Discriminator.
  std::cout << "Calling Discriminator Gradient:" << std::endl;
  discriminator.Gradient(discriminator.parameter, i, gradientDiscriminator,
      batchSize);
  //std::cout << "Updated Discriminator Gradient:" << std::endl << gradientDiscriminator << std::endl;

  noise.imbue( [&]() { return noiseFunction();} );
  //std::cout << "Noise:" << std::endl << noise << std::endl;
  std::cout << "Calling Generator Forward:" << std::endl;
  generator.Forward(std::move(noise));
  //std::cout << "Discriminator Predictors:" << std::endl << discriminator.predictors << std::endl;
  discriminator.predictors.col(numFunctions) = boost::apply_visitor(
      outputParameterVisitor, generator.network.back());
  //std::cout << "Updated Discriminator Predictors:" << std::endl << discriminator.predictors << std::endl;

  discriminator.responses(numFunctions) = 0;
  //std::cout << "Discriminator Responses:" << std::endl << discriminator.responses << std::endl;
  //std::cout << "Noise Discriminator Gradient:" << std::endl << noiseGradientDiscriminator << std::endl;
  std::cout << "Calling Discriminator Gradient:" << std::endl;
  discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator, batchSize);
  //std::cout << "Updated Noise Discriminator Gradient:" << std::endl << noiseGradientDiscriminator << std::endl;
  //std::cout << "Discriminator Gradient:" << std::endl << gradientDiscriminator << std::endl;
  gradientDiscriminator += noiseGradientDiscriminator;
  //std::cout << "Updated Discriminator Gradient:" << std::endl << gradientDiscriminator << std::endl;

  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(noise))).
    // Pass the error from Discriminator to Generator.
    std::cout << "Minimizing Error Objective:" << std::endl;
    //std::cout << "Discriminator Responses:" << std::endl << discriminator.responses << std::endl;
    discriminator.responses(numFunctions) = 1;
    //std::cout << "Updated Discriminator Responses:" << std::endl << discriminator.responses << std::endl;
    //std::cout << "Noise Discriminator Gradient:" << std::endl << noiseGradientDiscriminator << std::endl;
    std::cout << "Calling Discriminator Gradient:" << std::endl;
    discriminator.Gradient(discriminator.parameter, numFunctions,
      noiseGradientDiscriminator, batchSize);
    //std::cout << "Updated Noise Discriminator Gradient:" << std::endl << noiseGradientDiscriminator << std::endl;
    generator.error = boost::apply_visitor(deltaVisitor,
        discriminator.network[1]);
    //std::cout << "Generator Error:" << std::endl << generator.error << std::endl;
    //std::cout << "Noise:" << std::endl << noise << std::endl;
    //generator.currentInput = predictors;
    //std::cout << "Calling Generator Backward:" << std::endl;
    //generator.Backward();
    //std::cout << "Generator Gradient:" << std::endl << gradientGenerator << std::endl;
    std::cout << "Setting Predictors:" << std::endl;
    generator.Predictors() = noise;
    std::cout << "Generator Reset Gradients:" << std::endl;
    generator.ResetGradients(gradientGenerator);
    std::cout << "Calling Generator Gradient:" << std::endl;
    generator.Gradient(generator.parameter, 0, gradientGenerator, noise.n_cols);
    //std::cout << "Updated Generator Gradient:" << std::endl << gradientGenerator << std::endl;
    //double multiplier = 1.5;

    gradientGenerator *= multiplier;
    std::cout << "Updated Generator Gradient:" << std::endl << gradientGenerator << std::endl;
  }
  std::cout << "Counter:" << std::endl << counter << std::endl;
  counter++;
  std::cout << "Updated Counter:" << std::endl << counter << std::endl;
  std::cout << "Number of Functions:" << std::endl << numFunctions << std::endl;
  if (counter >= numFunctions)
  {
    std::cout << "Counter Reset:" << std::endl;
    counter = 0;
    currentBatch++;
    std::cout << "Current Batch: " << currentBatch << std::endl;
  }
  else if (counter % batchSize == 0)
  {
    std::cout << "Current Batch Update:" << std::endl;
    currentBatch++;
    std::cout << "Current Batch: " << currentBatch << std::endl;
    if (preTrainSize > 0)
    {
      preTrainSize--;
      std::cout << "Updated Pre Training Size: " << preTrainSize << std::endl;
    }
  }
  std::cout << "Exit GAN Gradient" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::Shuffle()
{
  std::cout << "Enter GAN Shuffle:" << std::endl;
  //std::cout << "Predictors:" << std::endl << predictors << std::endl;
  //std::cout << "Responses:" << std::endl << responses << std::endl;
  math::ShuffleData(predictors, responses, predictors, responses);
  //std::cout << "Updated Predictors:" << std::endl << predictors << std::endl;
  //std::cout << "Updated Responses:" << std::endl << responses << std::endl;
  std::cout << "Exit GAN Shuffle:" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::Forward(arma::mat&& input)
{
  std::cout << "Enter GAN Forward:" << std::endl;
  if (!reset)
    Reset();

  std::cout << "Input:" << std::endl << input << std::endl;
  std::cout << "Calling Generator Forward:" << std::endl;
  generator.Forward(std::move(input));
  //std::cout << "GAN Output:" << std::endl << ganOutput << std::endl;
  ganOutput = boost::apply_visitor(
      outputParameterVisitor,
      generator.network.back());
  std::cout << "Updated GAN Output:" << std::endl << ganOutput << std::endl;
  std::cout << "Calling Discriminator Forward:" << std::endl;
  discriminator.Forward(std::move(ganOutput));
  std::cout << "Exit GAN Forward:" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
void GAN<Model, InitializationRuleType, Noise>::
Predict(arma::mat&& input, arma::mat& output)
{
  std::cout << "Enter GAN Predict:" << std::endl;
  if (!reset)
    Reset();

  std::cout << "Input:" << std::endl << input << std::endl;
  std::cout << "Calling GAN Forward:" << std::endl;
  Forward(std::move(input));

  //std::cout << "Output:" << std::endl << output << std::endl;
  output = boost::apply_visitor(outputParameterVisitor,
      discriminator.network.back());
  std::cout << "Predicted Output:" << std::endl << output << std::endl;
  std::cout << "Exit GAN Predict:" << std::endl;
}

template<typename Model, typename InitializationRuleType, typename Noise>
template<typename Archive>
void GAN<Model, InitializationRuleType, Noise>::
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
