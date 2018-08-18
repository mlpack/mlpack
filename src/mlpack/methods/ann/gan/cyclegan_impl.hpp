/**
 * @file cyclegan_impl.hpp
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_CYCLEGAN_IMPL_HPP
#define MLPACK_METHODS_ANN_GAN_CYCLEGAN_IMPL_HPP

#include "cyclegan.hpp"

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
  typename InitializationRuleType
>
CycleGAN<Model, InitializationRuleType>::CycleGAN(
    arma::mat& trainDataX,
    arma::mat& trainDataY,
    Model generatorX,
    Model generatorY,
    Model discriminatorX,
    Model discriminatorY,
    InitializationRuleType& initializeRule,
    const size_t batchSize,
    const size_t generatorUpdateStep,
    const size_t preTrainSize,
    const double lambda,
    const double multiplier):
    predictorsX(trainDataX),
    predictorsY(trainDataY),
    generatorX(std::move(generatorX)),
    generatorY(std::move(generatorY)),
    discriminatorX(std::move(discriminatorX)),
    discriminatorY(std::move(discriminatorY)),
    initializeRule(initializeRule),
    batchSize(batchSize),
    generatorUpdateStep(generatorUpdateStep),
    preTrainSize(preTrainSize),
    lambda(lambda),
    multiplier(multiplier),
    reset(false)
{
  // Insert IdentityLayer for joining the generatorY and discriminatorX.
  this->discriminatorX.network.insert(this->discriminatorX.network.begin(),
      new IdentityLayer<>());

  // Insert IdentityLayer for joining the generatorX and discriminatorY.
  this->discriminatorY.network.insert(this->discriminatorY.network.begin(),
      new IdentityLayer<>());

  counterX = counterY = 0;
  currentBatch = 0;

  this->discriminatorX.deterministic = this->generatorY.deterministic = true;
  this->discriminatorY.deterministic = this->generatorX.deterministic = true;

  responsesX.set_size(1, trainDataX.n_cols);
  responsesY.set_size(1, trainDataY.n_cols);
  responsesX.ones();
  responsesY.ones();

  this->discriminatorX.predictors.set_size(trainDataX.n_rows,
      trainDataX.n_cols + batchSize);
  this->discriminatorX.predictors.cols(0, trainDataX.n_cols - 1) = trainDataX;
  this->discriminatorX.responses.set_size(1, trainDataX.n_cols + batchSize);
  this->discriminatorX.responses.cols(trainDataX.n_cols,
      trainDataX.n_cols + batchSize - 1) = arma::zeros(1, batchSize);

  this->discriminatorY.predictors.set_size(trainDataY.n_rows,
      trainDataY.n_cols + batchSize);
  this->discriminatorY.predictors.cols(0, trainDataY.n_cols - 1) = trainDataY;
  this->discriminatorY.responses.set_size(1, trainDataY.n_cols + batchSize);
  this->discriminatorY.responses.cols(trainDataY.n_cols,
      trainDataY.n_cols + batchSize - 1) = arma::zeros(1, batchSize);

  numFunctionsX = trainDataX.n_cols;
  numFunctionsY = trainDataY.n_cols;

  this->generatorX.predictors.set_size(trainDataX.n_rows, batchSize);
  this->generatorX.responses.set_size(trainDataY.n_rows, batchSize);

  this->generatorY.predictors.set_size(trainDataY.n_rows, batchSize);
  this->generatorY.responses.set_size(trainDataX.n_rows, batchSize);
}

template<
  typename Model,
  typename InitializationRuleType
>
CycleGAN<Model, InitializationRuleType>::CycleGAN(
    const CycleGAN& network):
    predictorsX(network.predictorsX),
    predictorsY(network.predictorsY),
    responsesX(network.responsesX),
    responsesY(network.responsesY),
    generatorX(network.generatorX),
    generatorY(network.generatorY),
    discriminatorX(network.discriminatorX),
    discriminatorY(network.discriminatorY),
    initializeRule(network.initializeRule),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    lambda(network.lambda),
    multiplier(multiplier),
    reset(network.reset),
    counterX(network.counterX),
    counterY(network.counterY),
    currentBatch(network.currentBatch),
    parameter(network.parameter),
    numFunctionsX(network.numFunctionsX),
    numFunctionsY(network.numFunctionsY)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType
>
CycleGAN<Model, InitializationRuleType>::CycleGAN(
    CycleGAN&& network):
    predictorsX(std::move(network.predictorsX)),
    predictorsY(std::move(network.predictorsY)),
    responsesX(std::move(network.responsesX)),
    responsesY(std::move(network.responsesY)),
    generatorX(std::move(network.generatorX)),
    generatorY(std::move(network.generatorY)),
    discriminatorX(std::move(network.discriminatorX)),
    discriminatorY(std::move(network.discriminatorY)),
    initializeRule(std::move(network.initializeRule)),
    batchSize(network.batchSize),
    generatorUpdateStep(network.generatorUpdateStep),
    preTrainSize(network.preTrainSize),
    lambda(network.lambda),
    multiplier(multiplier),
    reset(network.reset),
    counterX(network.counterX),
    counterY(network.counterY),
    currentBatch(network.currentBatch),
    parameter(std::move(network.parameter)),
    numFunctionsX(network.numFunctionsX),
    numFunctionsY(network.numFunctionsY)
{
  /* Nothing to do here */
}

template<
  typename Model,
  typename InitializationRuleType
>
void CycleGAN<Model, InitializationRuleType>::Reset()
{
  size_t genWeightsX = 0;
  size_t discWeightsX = 0;
  size_t genWeightsY = 0;
  size_t discWeightsY = 0;

  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);

  for (size_t i = 0; i < generatorX.network.size(); ++i)
  {
    genWeightsX += boost::apply_visitor(weightSizeVisitor,
        generatorX.network[i]);
  }

  for (size_t i = 0; i < discriminatorX.network.size(); ++i)
  {
    discWeightsX += boost::apply_visitor(weightSizeVisitor,
        discriminatorX.network[i]);
  }

  for (size_t i = 0; i < generatorY.network.size(); ++i)
  {
    genWeightsY += boost::apply_visitor(weightSizeVisitor,
        generatorY.network[i]);
  }

  for (size_t i = 0; i < discriminatorY.network.size(); ++i)
  {
    discWeightsY += boost::apply_visitor(weightSizeVisitor,
        discriminatorY.network[i]);
  }

  parameter.set_size(genWeightsX + discWeightsX + genWeightsY + discWeightsY,
      1);
  generatorX.Parameters() = arma::mat(parameter.memptr(), genWeightsX, 1, false,
      false);
  discriminatorX.Parameters() = arma::mat(parameter.memptr() + genWeightsX,
      discWeightsX, 1, false, false);
  generatorY.Parameters() = arma::mat(parameter.memptr() + genWeightsX +
      discWeightsX, genWeightsY, 1, false, false);
  discriminatorY.Parameters() = arma::mat(parameter.memptr() + genWeightsX +
      discWeightsX + genWeightsY, discWeightsY, 1, false, false);

  // Initialize the parameters generatorX.
  networkInit.Initialize(generatorX.network, parameter);
  // Initialize the parameters discriminatorX.
  networkInit.Initialize(discriminatorX.network, parameter, genWeightsX);
  // Initialize the parameters generatorY.
  networkInit.Initialize(generatorY.network, parameter, genWeightsX +
      discWeightsX);
  // Initialize the parameters discriminatorY.
  networkInit.Initialize(discriminatorY.network, parameter, genWeightsX +
      discWeightsX + genWeightsY);

  reset = true;
}

template<
  typename Model,
  typename InitializationRuleType
>
template<typename OptimizerType>
void CycleGAN<Model, InitializationRuleType>::Train(
    OptimizerType& Optimizer)
{
  if (!reset)
  {
    Reset();
  }

  return Optimizer.Optimize(*this, parameter);
}

template<
  typename Model,
  typename InitializationRuleType
>
double CycleGAN<Model, InitializationRuleType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t i,
    const size_t /* batchSize */)
{
  if (!reset)
  {
    Reset();
  }

  // Passing real image X into discriminatorX.
  currentInput = arma::mat(predictorsX.memptr() + (i * predictorsX.n_rows),
      predictorsX.n_rows, batchSize, false, false);
  currentTarget = arma::mat(responsesX.memptr() + i, 1, batchSize, false,
      false);
  discriminatorX.Forward(std::move(currentInput));
  double res = discriminatorX.outputLayer.Forward(std::move(
      boost::apply_visitor(outputParameterVisitor,
      discriminatorX.network.back())), std::move(currentTarget));

  // Passing generated image Y into discriminatorY.
  generatorX.Forward(std::move(currentInput));
  discriminatorY.predictors.cols(numFunctionsY, numFunctionsY + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generatorX.network.back());
  discriminatorY.Forward(std::move(discriminatorY.predictors.cols(numFunctionsY,
      numFunctionsY + batchSize - 1)));
  discriminatorY.responses.cols(numFunctionsY, numFunctionsY + batchSize - 1) =
      arma::zeros(1, batchSize);
  currentInput = arma::mat(discriminatorY.predictors.memptr() + (numFunctionsY *
      discriminatorY.predictors.n_rows), discriminatorY.predictors.n_rows,
      batchSize, false, false);
  currentTarget = arma::mat(discriminatorY.responses.memptr() + numFunctionsY,
      1, batchSize, false, false);
  res += discriminatorY.outputLayer.Forward(std::move(boost::apply_visitor(
      outputParameterVisitor, discriminatorY.network.back())),
      std::move(currentTarget));

  // Computing the regularization between the real image X and cyclic X.
  generatorY.Forward(std::move(currentInput));
  discriminatorX.predictors.cols(numFunctionsX, numFunctionsX + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generatorY.network.back());
  currentInput = arma::mat(predictorsX.memptr() + (i * predictorsX.n_rows),
      predictorsX.n_rows, batchSize, false, false);
  res += lambda * norm(discriminatorX.predictors.cols(numFunctionsX,
      numFunctionsX + batchSize - 1) - currentInput, 1);

  // Passing real image Y into discriminatorY.
  currentInput = arma::mat(predictorsY.memptr() + (i * predictorsY.n_rows),
      predictorsY.n_rows, batchSize, false, false);
  currentTarget = arma::mat(responsesY.memptr() + i, 1, batchSize, false,
      false);
  discriminatorY.Forward(std::move(currentInput));
  res += discriminatorY.outputLayer.Forward(
      std::move(boost::apply_visitor(
      outputParameterVisitor,
      discriminatorY.network.back())), std::move(currentTarget));

  // Passing generated image X into discriminatorX.
  generatorY.Forward(std::move(currentInput));
  discriminatorX.predictors.cols(numFunctionsX, numFunctionsX + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generatorY.network.back());
  discriminatorX.Forward(std::move(discriminatorX.predictors.cols(numFunctionsX,
      numFunctionsX + batchSize - 1)));
  discriminatorX.responses.cols(numFunctionsX, numFunctionsX + batchSize - 1) =
      arma::zeros(1, batchSize);
  currentInput = arma::mat(discriminatorX.predictors.memptr() + (numFunctionsX *
      discriminatorX.predictors.n_rows), discriminatorX.predictors.n_rows,
      batchSize, false, false);
  currentTarget = arma::mat(discriminatorX.responses.memptr() + numFunctionsX,
      1, batchSize, false, false);
  res += discriminatorX.outputLayer.Forward(std::move(boost::apply_visitor(
      outputParameterVisitor, discriminatorX.network.back())),
      std::move(currentTarget));

  // Computing the regularization between the real image Y and cyclic Y.
  generatorX.Forward(std::move(currentInput));
  discriminatorY.predictors.cols(numFunctionsY, numFunctionsY + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generatorX.network.back());
  currentInput = arma::mat(predictorsY.memptr() + (i * predictorsY.n_rows),
      predictorsY.n_rows, batchSize, false, false);
  res += lambda * norm(discriminatorY.predictors.cols(numFunctionsY,
      numFunctionsY + batchSize - 1) - currentInput, 1);

  return res;
}

template<
  typename Model,
  typename InitializationRuleType
>
template<typename GradType>
double CycleGAN<Model, InitializationRuleType>::EvaluateWithGradient(
    const arma::mat& /* parameters */,
    const size_t i,
    GradType& gradient,
    const size_t /* batchSize */)
{
  if (!reset)
  {
    Reset();
  }

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

  if (generatedGradientDX.is_empty() || generatedGradientDY.is_empty() ||
      generatedGradientGX.is_empty() || generatedGradientGY.is_empty())
  {
    generatedGradientDX = arma::zeros<arma::mat>(
        gradientDiscriminatorX.n_elem, 1);
    generatedGradientDY = arma::zeros<arma::mat>(
        gradientDiscriminatorY.n_elem, 1);
    generatedGradientGX = arma::zeros<arma::mat>(
        gradientGeneratorX.n_elem, 1);
    generatedGradientGY = arma::zeros<arma::mat>(
        gradientGeneratorY.n_elem, 1);
  }
  else
  {
    generatedGradientDX.zeros();
    generatedGradientDY.zeros();
    generatedGradientGX.zeros();
    generatedGradientGY.zeros();
  }

  gradientGeneratorX = arma::mat(gradient.memptr(),
      generatorX.Parameters().n_elem, 1, false, false);

  gradientDiscriminatorX = arma::mat(gradient.memptr() +
      gradientGeneratorX.n_elem,
      discriminatorX.Parameters().n_elem, 1, false, false);

  gradientGeneratorY = arma::mat(gradient.memptr() + gradientGeneratorX.n_elem
      + gradientDiscriminatorX.n_elem, generatorY.Parameters().n_elem, 1,
      false, false);

  gradientDiscriminatorY = arma::mat(gradient.memptr() +
      gradientGeneratorX.n_elem + gradientDiscriminatorX.n_elem +
      gradientGeneratorY.n_elem, discriminatorY.Parameters().n_elem, 1, false,
      false);

  // Get the gradients from the real data X and Y.
  double res = discriminatorX.EvaluateWithGradient(discriminatorX.parameter,
      i, gradientDiscriminatorX, batchSize);
  res += discriminatorY.EvaluateWithGradient(discriminatorY.parameter,
      i, gradientDiscriminatorY, batchSize);

  // Generate data from X and Y.
  currentInput = arma::mat(predictorsX.memptr() + (i * predictorsX.n_rows),
      predictorsX.n_rows, batchSize, false, false);
  generatorX.Forward(std::move(currentInput));
  arma::mat generatedY = boost::apply_visitor(outputParameterVisitor,
      generatorX.network.back());
  currentInput = arma::mat(predictorsY.memptr() + (i * predictorsY.n_rows),
      predictorsY.n_rows, batchSize, false, false);
  generatorY.Forward(std::move(currentInput));
  arma::mat generatedX = boost::apply_visitor(outputParameterVisitor,
      generatorY.network.back());
  discriminatorY.predictors.cols(numFunctionsY, numFunctionsY + batchSize - 1) =
      generatedY;
  discriminatorX.predictors.cols(numFunctionsX, numFunctionsX + batchSize - 1) =
      generatedX;
  discriminatorY.responses.cols(numFunctionsY, numFunctionsY + batchSize - 1) =
      arma::zeros(1, batchSize);
  discriminatorX.responses.cols(numFunctionsX, numFunctionsX + batchSize - 1) =
      arma::zeros(1, batchSize);

  // Get the gradients from the generated data.
  res += discriminatorY.EvaluateWithGradient(discriminatorY.parameter,
      numFunctionsY, generatedGradientDY, batchSize);
  res += discriminatorX.EvaluateWithGradient(discriminatorX.parameter,
      numFunctionsX, generatedGradientDX, batchSize);
  gradientDiscriminatorY += generatedGradientDY;
  gradientDiscriminatorX += generatedGradientDX;

  // Update the Generator gradients.
  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize -log(D(G(X))) and -log(D(G(Y))).
    // Pass the error from Discriminators to Generators.
    discriminatorX.responses.cols(numFunctionsX, numFunctionsX + batchSize - 1)
        = arma::ones(1, batchSize);
    discriminatorY.responses.cols(numFunctionsY, numFunctionsY + batchSize - 1)
        = arma::ones(1, batchSize);
    discriminatorX.Gradient(discriminatorX.parameter, numFunctionsX,
        generatedGradientDX, batchSize);
    discriminatorY.Gradient(discriminatorY.parameter, numFunctionsY,
        generatedGradientDY, batchSize);
    generatorY.error = boost::apply_visitor(deltaVisitor,
        discriminatorX.network[1]);
    generatorX.error = boost::apply_visitor(deltaVisitor,
        discriminatorY.network[1]);

    generatorY.Predictors() = arma::mat(predictorsY.memptr() + (i *
      predictorsY.n_rows), predictorsY.n_rows, batchSize, false, false);
    generatorX.Predictors() = arma::mat(predictorsX.memptr() + (i *
      predictorsX.n_rows), predictorsX.n_rows, batchSize, false, false);
    generatorY.ResetGradients(gradientGeneratorY);
    generatorX.ResetGradients(gradientGeneratorX);
    generatorY.Gradient(generatorY.parameter, 0, gradientGeneratorY, batchSize);
    generatorX.Gradient(generatorX.parameter, 0, gradientGeneratorX, batchSize);
  }

  // Regenerate the original data back from generated data.
  generatorX.Forward(std::move(generatedX));
  generatorY.Forward(std::move(generatedY));
  discriminatorY.predictors.cols(numFunctionsY, numFunctionsY + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generatorX.network.back());
  discriminatorX.predictors.cols(numFunctionsX, numFunctionsX + batchSize - 1) =
      boost::apply_visitor(outputParameterVisitor, generatorY.network.back());

  // Update the Generator gradients.
  if (currentBatch % generatorUpdateStep == 0 && preTrainSize == 0)
  {
    // Minimize F(G(X)) and G(F(Y)).
    // Pass the error to the Generators.
    currentInput = arma::mat(predictorsY.memptr() + (i * predictorsY.n_rows),
        predictorsY.n_rows, batchSize, false, false);
    generatorY.error = arma::abs(discriminatorY.predictors.cols(numFunctionsY,
        numFunctionsY + batchSize - 1) - currentInput);
    // Add cyclic loss.
    res += lambda * arma::accu(generatorY.error);

    currentInput = arma::mat(predictorsX.memptr() + (i * predictorsX.n_rows),
        predictorsX.n_rows, batchSize, false, false);
    generatorX.error = arma::abs(discriminatorX.predictors.cols(numFunctionsX,
        numFunctionsX + batchSize - 1) - currentInput);
    // Add cyclic loss.
    res += lambda * arma::accu(generatorX.error);

    generatorY.Predictors() = generatedY;
    generatorX.Predictors() = generatedX;
    generatorY.ResetGradients(generatedGradientGY);
    generatorX.ResetGradients(generatedGradientGX);
    generatorY.Gradient(generatorY.parameter, 0, generatedGradientGY,
        batchSize);
    generatorX.Gradient(generatorX.parameter, 0, generatedGradientGX,
        batchSize);

    gradientGeneratorY += generatedGradientGY;
    gradientGeneratorX += generatedGradientGX;
    gradientGeneratorY *= multiplier;
    gradientGeneratorX *= multiplier;
  }

  counterX++;
  counterY++;
  currentBatch++;

  // Revert the counter to zero, if the total dataset get's covered.
  if (counterX * batchSize >= numFunctionsX)
  {
    counterX = 0;
  }
  if (counterY * batchSize >= numFunctionsY)
  {
    counterY = 0;
  }

  if (preTrainSize > 0)
  {
    preTrainSize--;
  }

  return res;
}

template<
  typename Model,
  typename InitializationRuleType
>
void CycleGAN<Model, InitializationRuleType>::Gradient(
    const arma::mat& parameters,
    const size_t i,
    arma::mat& gradient,
    const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, i, gradient, batchSize);
}

template<
  typename Model,
  typename InitializationRuleType
>
void CycleGAN<Model, InitializationRuleType>::Shuffle()
{
  math::ShuffleData(predictorsX, responsesX, predictorsX, responsesX);
  math::ShuffleData(predictorsY, responsesY, predictorsY, responsesY);
}

template<
  typename Model,
  typename InitializationRuleType
>
void CycleGAN<Model, InitializationRuleType>::Forward(
    arma::mat&& input,
    const bool xtoy)
{
  if (!reset)
  {
    Reset();
  }

  if (xtoy)
  {
    generatorX.Forward(std::move(input));
    ganOutput = boost::apply_visitor(outputParameterVisitor,
        generatorX.network.back());

    discriminatorY.Forward(std::move(ganOutput));
  }
  else
  {
    generatorY.Forward(std::move(input));
    ganOutput = boost::apply_visitor(outputParameterVisitor,
        generatorY.network.back());

    discriminatorX.Forward(std::move(ganOutput));
  }
}

template<
  typename Model,
  typename InitializationRuleType
>
void CycleGAN<Model, InitializationRuleType>::Predict(
    arma::mat&& input,
    arma::mat& output,
    const bool xtoy)
{
  if (!reset)
  {
    Reset();
  }

  Forward(std::move(input), xtoy);

  if (xtoy)
  {
    output = boost::apply_visitor(outputParameterVisitor,
        discriminatorY.network.back());
  }
  else
  {
    output = boost::apply_visitor(outputParameterVisitor,
        discriminatorX.network.back());
  }
}

template<
  typename Model,
  typename InitializationRuleType
>
template<typename Archive>
void CycleGAN<Model, InitializationRuleType>::
serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(generatorX);
  ar & BOOST_SERIALIZATION_NVP(generatorY);
  ar & BOOST_SERIALIZATION_NVP(discriminatorX);
  ar & BOOST_SERIALIZATION_NVP(discriminatorY);
}

} // namespace ann
} // namespace mlpack
# endif
