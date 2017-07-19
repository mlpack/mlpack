/**
 * @file rnn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the RNN class, which implements recurrent neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RNN_IMPL_HPP
#define MLPACK_METHODS_ANN_RNN_IMPL_HPP

// In case it hasn't been included yet.
#include "rnn.hpp"

#include "visitor/load_output_parameter_visitor.hpp"
#include "visitor/save_output_parameter_visitor.hpp"
#include "visitor/forward_visitor.hpp"
#include "visitor/backward_visitor.hpp"
#include "visitor/deterministic_set_visitor.hpp"
#include "visitor/gradient_set_visitor.hpp"
#include "visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
RNN<OutputLayerType, InitializationRuleType, MatrixType>::RNN(
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    rho(rho),
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single)
{
  /* Nothing to do here */
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
RNN<OutputLayerType, InitializationRuleType, MatrixType>::RNN(
    MatrixType predictors,
    MatrixType responses,
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    rho(rho),
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single),
    predictors(std::move(predictors)),
    responses(std::move(responses)),
    deterministic(true)
{
  numFunctions = this->responses.n_cols;
  ResetDeterministic();
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
RNN<OutputLayerType, InitializationRuleType, MatrixType>::~RNN()
{
  for (LayerTypes& layer : network)
  {
    boost::apply_visitor(deleteVisitor, layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
template<typename OptimizerType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Train(
    MatrixType predictors,
    MatrixType responses,
    OptimizerType& optimizer)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
    reset = true;
  }

  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
template<typename OptimizerType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Train(
    MatrixType predictors, MatrixType responses)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
    reset = true;
  }

  OptimizerType optimizer;

  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Predict(
    MatrixType predictors, MatrixType& results)
{
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  results = arma::zeros<MatrixType>(outputSize * rho, predictors.n_cols);
  MatrixType resultsTemp = results.col(0);

  for (size_t i = 0; i < predictors.n_cols; i++)
  {
    SinglePredict(
        MatrixType(predictors.colptr(i), predictors.n_rows, 1, false, true),
        resultsTemp);

    results.col(i) = resultsTemp;
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::SinglePredict(
    const MatrixType& predictors, MatrixType& results)
{
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentInput = predictors.rows(seqNum * inputSize,
        (seqNum + 1) * inputSize - 1);
    Forward(std::move(currentInput));

    results.rows(seqNum * outputSize, (seqNum + 1) * outputSize - 1) =
        boost::apply_visitor(outputParameterVisitor, network.back());
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
double RNN<OutputLayerType, InitializationRuleType, MatrixType>::Evaluate(
    const MatrixType& /* parameters */, const size_t i, const bool deterministic)
{
  if (parameter.is_empty())
  {
    ResetParameters();
    reset = true;
  }

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  MatrixType input = MatrixType(predictors.colptr(i), predictors.n_rows,
      1, false, true);
  MatrixType target = MatrixType(responses.colptr(i), responses.n_rows,
      1, false, true);

  if (!inputSize)
  {
    inputSize = input.n_elem / rho;
    targetSize = target.n_elem / rho;
  }

  double performance = 0;

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentInput = input.rows(seqNum * inputSize, (seqNum + 1) * inputSize - 1);
    MatrixType currentTarget = target.rows(seqNum * targetSize,
        (seqNum + 1) * targetSize - 1);

    Forward(std::move(currentInput));

    if (!deterministic)
    {
      for (size_t l = 0; l < network.size(); ++l)
      {
        boost::apply_visitor(SaveOutputParameterVisitor(
            std::move(moduleOutputParameter)), network[l]);
      }
    }

    performance += outputLayer.Forward(std::move(boost::apply_visitor(
        outputParameterVisitor, network.back())), std::move(currentTarget));
  }

  if (!outputSize)
  {
    outputSize = boost::apply_visitor(outputParameterVisitor,
        network.back()).n_elem;
  }

  return performance;
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Gradient(
    const MatrixType& parameters, const size_t i, MatrixType& gradient)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      ResetParameters();
      reset = true;
    }

    gradient = arma::zeros<MatrixType>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  Evaluate(parameters, i, false);

  MatrixType currentGradient = arma::zeros<MatrixType>(parameter.n_rows,
      parameter.n_cols);
  ResetGradients(currentGradient);

  MatrixType input = MatrixType(predictors.colptr(i), predictors.n_rows,
      1, false, true);
  MatrixType target = MatrixType(responses.colptr(i), responses.n_rows,
      1, false, true);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentGradient.zeros();

    MatrixType currentTarget = target.rows((rho - seqNum - 1) * targetSize,
        (rho - seqNum) * targetSize - 1);
    currentInput = input.rows((rho - seqNum - 1) * inputSize,
        (rho - seqNum) * inputSize - 1);

    for (size_t l = 0; l < network.size(); ++l)
    {
      boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(moduleOutputParameter)), network[network.size() - 1 - l]);
    }

    if (single && seqNum > 0)
    {
      error.zeros();
    }
    else
    {
      outputLayer.Backward(std::move(boost::apply_visitor(
          outputParameterVisitor, network.back())), std::move(currentTarget),
          std::move(error));
    }

    Backward();
    Gradient();
    gradient += currentGradient;
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::ResetParameters()
{
  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network, parameter);
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::ResetDeterministic()
{
  DeterministicSetVisitor deterministicSetVisitor(deterministic);
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deterministicSetVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::ResetGradients(
    MatrixType& gradient)
{
  size_t offset = 0;
  for (LayerTypes& layer : network)
  {
    offset += boost::apply_visitor(GradientSetVisitor(std::move(gradient),
        offset), layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Forward(MatrixType&& input)
{
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, network.front()))),
      network.front());

  for (size_t i = 1; i < network.size(); ++i)
  {
    boost::apply_visitor(ForwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network[i - 1])),
        std::move(boost::apply_visitor(outputParameterVisitor, network[i]))),
        network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Backward()
{
  boost::apply_visitor(BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network.back())),
        std::move(error), std::move(boost::apply_visitor(deltaVisitor,
        network.back()))), network.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    boost::apply_visitor(BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
        network[network.size() - i])), std::move(boost::apply_visitor(
        deltaVisitor, network[network.size() - i + 1])), std::move(
        boost::apply_visitor(deltaVisitor, network[network.size() - i]))),
        network[network.size() - i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Gradient()
{
  boost::apply_visitor(GradientVisitor(std::move(currentInput), std::move(
      boost::apply_visitor(deltaVisitor, network[1]))), network.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    boost::apply_visitor(GradientVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network[i - 1])),
        std::move(boost::apply_visitor(deltaVisitor, network[i + 1]))),
        network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType, typename MatrixType>
template<typename Archive>
void RNN<OutputLayerType, InitializationRuleType, MatrixType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(rho, "rho");
  ar & data::CreateNVP(single, "single");
  ar & data::CreateNVP(inputSize, "inputSize");
  ar & data::CreateNVP(outputSize, "outputSize");
  ar & data::CreateNVP(targetSize, "targetSize");
  ar & data::CreateNVP(currentInput, "currentInput");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    reset = false;

    size_t offset = 0;
    for (LayerTypes& layer : network)
    {
      offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
          offset), layer);

      boost::apply_visitor(resetVisitor, layer);
    }
  }
}

} // namespace ann
} // namespace mlpack

#endif
