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
#include "visitor/reset_cell_visitor.hpp"
#include "visitor/deterministic_set_visitor.hpp"
#include "visitor/gradient_set_visitor.hpp"
#include "visitor/gradient_visitor.hpp"
#include "visitor/weight_set_visitor.hpp"

#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType, typename InitializationRuleType>
RNN<OutputLayerType, InitializationRuleType>::RNN(
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    rho(rho),
    prevRho(0),
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single),
    numFunctions(0),
    deterministic(true)
{
  /* Nothing to do here */
}

template<typename OutputLayerType, typename InitializationRuleType>
RNN<OutputLayerType, InitializationRuleType>::RNN(
    arma::mat predictors,
    arma::mat responses,
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    rho(rho),
    prevRho(0),
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single),
    predictors(std::move(predictors)),
    responses(std::move(responses)),
    numFunctions(0),
    deterministic(true)
{
  numFunctions = this->responses.n_cols;
  ResetDeterministic();
}

template<typename OutputLayerType, typename InitializationRuleType>
RNN<OutputLayerType, InitializationRuleType>::~RNN()
{
  for (LayerTypes& layer : network)
  {
    boost::apply_visitor(deleteVisitor, layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename OptimizerType>
void RNN<OutputLayerType, InitializationRuleType>::Train(
    arma::mat predictors,
    arma::mat responses,
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

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::ResetCells()
{
  for (size_t i = 1; i < network.size(); ++i)
  {
    boost::apply_visitor(ResetCellVisitor(), network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename OptimizerType>
void RNN<OutputLayerType, InitializationRuleType>::Train(
    arma::mat predictors, arma::mat responses)
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

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::Predict(
    arma::mat predictors, arma::mat& results)
{
  ResetCells();

  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  results = arma::zeros<arma::mat>(outputSize * rho, predictors.n_cols);
  arma::mat resultsTemp = results.col(0);

  for (size_t i = 0; i < predictors.n_cols; i++)
  {
    SinglePredict(
        arma::mat(predictors.colptr(i), predictors.n_rows, 1, false, true),
        resultsTemp);

    results.col(i) = resultsTemp;
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::SinglePredict(
    const arma::mat& predictors, arma::mat& results)
{
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    Forward(std::move(predictors.rows(seqNum * inputSize,
        (seqNum + 1) * inputSize - 1)));

    results.rows(seqNum * outputSize, (seqNum + 1) * outputSize - 1) =
        boost::apply_visitor(outputParameterVisitor, network.back());
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
double RNN<OutputLayerType, InitializationRuleType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t begin,
    const size_t batchSize,
    const bool deterministic)
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

  if (!inputSize)
  {
    inputSize = predictors.n_rows / rho;
    targetSize = responses.n_rows / rho;
  }
  else if (targetSize == 0)
  {
    targetSize = responses.n_rows / rho;
  }

  ResetCells();

  double performance = 0;

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    Forward(std::move(predictors.submat(seqNum * inputSize, begin,
        (seqNum + 1) * inputSize - 1, begin + batchSize - 1)));

    if (!deterministic)
    {
      for (size_t l = 0; l < network.size(); ++l)
      {
        boost::apply_visitor(SaveOutputParameterVisitor(
            std::move(moduleOutputParameter)), network[l]);
      }
    }

    performance += outputLayer.Forward(std::move(boost::apply_visitor(
        outputParameterVisitor, network.back())),
        std::move(responses.submat(seqNum * targetSize, begin,
        (seqNum + 1) * targetSize - 1, begin + batchSize - 1)));
  }

  if (outputSize == 0)
  {
    outputSize = boost::apply_visitor(outputParameterVisitor,
        network.back()).n_elem / batchSize;
  }

  return performance;
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::Gradient(
    const arma::mat& parameters,
    const size_t begin,
    arma::mat& gradient,
    const size_t batchSize)
{
  // Initialize passed gradient.
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      ResetParameters();
      reset = true;
    }

    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  Evaluate(parameters, begin, batchSize, false);

  // Initialize current/working gradient.
  if (currentGradient.is_empty())
  {
    currentGradient = arma::zeros<arma::mat>(parameter.n_rows,
        parameter.n_cols);
  }

  ResetGradients(currentGradient);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentGradient.zeros();

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
          outputParameterVisitor, network.back())),
          std::move(responses.submat((rho - seqNum - 1) * targetSize, begin,
          (rho - seqNum) * targetSize - 1, begin + batchSize - 1)),
          std::move(error));
    }

    Backward();
    Gradient(std::move(predictors.submat((rho - seqNum - 1) * inputSize, begin,
        (rho - seqNum) * inputSize - 1, begin + batchSize - 1)));
    gradient += currentGradient;
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::ResetParameters()
{
  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network, parameter);
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::ResetDeterministic()
{
  DeterministicSetVisitor deterministicSetVisitor(deterministic);
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deterministicSetVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::ResetGradients(
    arma::mat& gradient)
{
  size_t offset = 0;
  for (LayerTypes& layer : network)
  {
    offset += boost::apply_visitor(GradientSetVisitor(std::move(gradient),
        offset), layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::Forward(arma::mat&& input)
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

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::Backward()
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

template<typename OutputLayerType, typename InitializationRuleType>
template<typename InputType>
void RNN<OutputLayerType, InitializationRuleType>::Gradient(InputType&& input)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(
      boost::apply_visitor(deltaVisitor, network[1]))), network.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    boost::apply_visitor(GradientVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network[i - 1])),
        std::move(boost::apply_visitor(deltaVisitor, network[i + 1]))),
        network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename Archive>
void RNN<OutputLayerType, InitializationRuleType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(rho, "rho");
  ar & data::CreateNVP(single, "single");
  ar & data::CreateNVP(inputSize, "inputSize");
  ar & data::CreateNVP(outputSize, "outputSize");
  ar & data::CreateNVP(targetSize, "targetSize");

  if (Archive::is_loading::value)
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
    network.clear();
  }

  ar & BOOST_SERIALIZATION_NVP(network);

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

    deterministic = true;
    ResetDeterministic();
  }
}

} // namespace ann
} // namespace mlpack

#endif
