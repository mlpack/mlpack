/**
 * @file methods/ann/ffn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the FFNEQL class, which implements feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_IMPL_HPP
#define MLPACK_METHODS_ANN_FFN_IMPL_HPP

// In case it hasn't been included yet.
#include "ffn_eql.hpp"

#include "visitor/forward_visitor.hpp"
#include "visitor/backward_visitor.hpp"
#include "visitor/deterministic_set_visitor.hpp"
#include "visitor/gradient_set_visitor.hpp"
#include "visitor/gradient_visitor.hpp"
#include "visitor/set_input_height_visitor.hpp"
#include "visitor/set_input_width_visitor.hpp"

#include "util/check_input_shape.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename typename InitializationRuleType,
         typename... CustomLayers>
FFNEQL<InitializationRuleType, CustomLayers...>::FFNEQL(
    InitializationRuleType initializeRule) :
    initializeRule(std::move(initializeRule)),
    width(0),
    height(0),
    reset(false),
    numFunctions(0),
    deterministic(false)
{
  /* Nothing to do here. */
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
FFNEQL<InitializationRuleType, CustomLayers...>::~FFNEQL()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType, CustomLayers...>::ResetData(
    arma::mat predictors, arma::mat responses)
{
  numFunctions = responses.n_cols;
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
  this->deterministic = false;
  ResetDeterministic();

  if (!reset)
    ResetParameters();
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType>
typename std::enable_if<
      HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
FFNEQL<InitializationRuleType, CustomLayers...>::
WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const
{
  if (optimizer.MaxIterations() < samples &&
      optimizer.MaxIterations() != 0)
  {
    Log::Warn << "The optimizer's maximum number of iterations "
              << "is less than the size of the dataset; the "
              << "optimizer will not pass over the entire "
              << "dataset. To fix this, modify the maximum "
              << "number of iterations to be at least equal "
              << "to the number of points of your dataset "
              << "(" << samples << ")." << std::endl;
  }
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType>
typename std::enable_if<
      !HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
FFNEQL<InitializationRuleType, CustomLayers...>::
WarnMessageMaxIterations(OptimizerType& /* optimizer */, size_t /* samples */)
    const
{
  return;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType, typename... CallbackTypes>
double FFNEQL<InitializationRuleType, CustomLayers...>::Train(
      arma::mat predictors,
      arma::mat responses,
      OptimizerType& optimizer,
      CallbackTypes&&... callbacks)
{
  CheckInputShape<std::vector<LayerTypes<CustomLayers...> > >(network,
                                                              predictors.n_rows,
                                                              "FFNEQL<>::Train()");

  ResetData(std::move(predictors), std::move(responses));

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameter, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFNEQL::FFNEQL(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType, typename... CallbackTypes>
double FFNEQL<InitializationRuleType, CustomLayers...>::Train(
    arma::mat predictors,
    arma::mat responses,
    CallbackTypes&&... callbacks)
{
  CheckInputShape<std::vector<LayerTypes<CustomLayers...> > >(network,
                                                              predictors.n_rows,
                                                              "FFNEQL<>::Train()");

  ResetData(std::move(predictors), std::move(responses));

  OptimizerType optimizer;

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameter, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFNEQL::FFNEQL(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename PredictorsType, typename ResponsesType, typename WeightsType>
void FFNEQL<InitializationRuleType, CustomLayers...>::Forward(
    const PredictorsType& inputs, ResponsesType& results,
    const WeightsType& weightSpace)
{
  if (parameter.is_empty())
    ResetParameters();

  Forward(inputs);
  results = boost::apply_visitor(outputParameterVisitor, network.back());
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename PredictorsType, typename ResponsesType>
void FFNEQL<InitializationRuleType, CustomLayers...>::Forward(
    const PredictorsType& inputs,
    ResponsesType& results,
    const size_t begin,
    const size_t end)
{
  boost::apply_visitor(ForwardVisitor(inputs,
      boost::apply_visitor(outputParameterVisitor, network[begin])),
      network[begin]);

  for (size_t i = 1; i < end - begin + 1; ++i)
  {
    boost::apply_visitor(ForwardVisitor(boost::apply_visitor(
        outputParameterVisitor, network[begin + i - 1]),
        boost::apply_visitor(outputParameterVisitor, network[begin + i])),
        network[begin + i]);
  }

  results = boost::apply_visitor(outputParameterVisitor, network[end]);
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename PredictorsType, typename TargetsType, typename GradientsType>
double FFNEQL<InitializationRuleType, CustomLayers...>::Backward(
    const PredictorsType& inputs,
    const TargetsType& targets,
    GradientsType& gradients,
    WeightsType& weightSpace)
{
  // Each preference vector is repeated batchSize * actionSize
  // number of times. Shape: (rewardSize, extendedSize * actionSize).
  const arma::mat extWeights = [&]()
  {
    arma::mat retval(rewardSize, extendedSize * actionSize);
    size_t colIdx {}, start {};
    size_t gap = batchSize * actionSize;

    while (colIdx < numWeights)
    {
      retval.submat(arma::span(0, rewardSize),
                    arma::span(start, start + gap - 1)) =
          arma::repmat(weightSpace.col(colIdx), 1, gap);
      start += gap;
      ++colIdx;
    }

    return retval;
  }();

  // Homotopy loss.
  const arma::mat prediction = boost::apply_visitor(outputParameterVisitor, network.back());
  const double numElem = arma::sum((prediction - target) != 0);

  const double lossA  =
      std::pow(arma::norm((prediction - target).vectorise()), 2) / numElem;
  const double lossB =
      std::pow(arma::norm(arma::sum(extWeights % (prediction - target))), 2) / numElem;

  const double homotopyLoss = (1 - lambda) * lossA + lambda * lossB;
  LambdaAnneal();

  // Gradient
  arma::mat errorA = (prediction - target) / numElem;
  arma::mat errorB = arma::sum(extWeights % (prediction - target)) % extWeights;

  error = 2 *((1 - lamdbda) * errorA + lambda * errorB);
  gradients = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);

  Backward();
  ResetGradients(gradients);
  Gradient(inputs);

  return homotopyLoss;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType, CustomLayers...>::Predict(
    arma::mat predictors, arma::mat& results)
{
  CheckInputShape<std::vector<LayerTypes<CustomLayers...> > >(
      network, predictors.n_rows, "FFNEQL<>::Predict()");

  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  arma::mat resultsTemp;
  Forward(arma::mat(predictors.colptr(0), predictors.n_rows, 1, false, true));
  resultsTemp = boost::apply_visitor(outputParameterVisitor,
      network.back()).col(0);

  results = arma::mat(resultsTemp.n_elem, predictors.n_cols);
  results.col(0) = resultsTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; ++i)
  {
    Forward(arma::mat(predictors.colptr(i), predictors.n_rows, 1, false, true));

    resultsTemp = boost::apply_visitor(outputParameterVisitor,
        network.back());
    results.col(i) = resultsTemp.col(0);
  }
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename PredictorsType, typename ResponsesType>
double FFNEQL<InitializationRuleType, CustomLayers...>::Evaluate(
    const PredictorsType& predictors, const ResponsesType& responses)
{
  CheckInputShape<std::vector<LayerTypes<CustomLayers...> > >(
      network, predictors.n_rows, "FFNEQL<>::Evaluate()");

  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  Forward(predictors);


  for (size_t i = 0; i < network.size(); ++i)
  {
    res += boost::apply_visitor(lossVisitor, network[i]);
  }

  return res;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
double FFNEQL<InitializationRuleType, CustomLayers...>::Evaluate(
    const arma::mat& parameters)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += Evaluate(parameters, i, 1, true);

  return res;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
double FFNEQL<InitializationRuleType, CustomLayers...>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t begin,
    const size_t batchSize,
    const bool deterministic)
{
  if (parameter.is_empty())
    ResetParameters();

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  Forward(predictors.cols(begin, begin + batchSize - 1));

  for (size_t i = 0; i < network.size(); ++i)
  {
    res += boost::apply_visitor(lossVisitor, network[i]);
  }

  return res;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
double FFNEQL<InitializationRuleType, CustomLayers...>::Evaluate(
    const arma::mat& parameters, const size_t begin, const size_t batchSize)
{
  return Evaluate(parameters, begin, batchSize, true);
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename GradType>
double FFNEQL<InitializationRuleType, CustomLayers...>::
EvaluateWithGradient(const arma::mat& parameters, GradType& gradient)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += EvaluateWithGradient(parameters, i, gradient, 1);

  return res;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename GradType>
double FFNEQL<InitializationRuleType, CustomLayers...>::
EvaluateWithGradient(const arma::mat& /* parameters */,
                     const size_t begin,
                     GradType& gradient,
                     const size_t batchSize)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
      ResetParameters();

    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  if (this->deterministic)
  {
    this->deterministic = false;
    ResetDeterministic();
  }

  Forward(predictors.cols(begin, begin + batchSize - 1));

  for (size_t i = 0; i < network.size(); ++i)
  {
    res += boost::apply_visitor(lossVisitor, network[i]);
  }

  Backward();
  ResetGradients(gradient);
  Gradient(predictors.cols(begin, begin + batchSize - 1));

  return res;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType, CustomLayers...>::Gradient(
    const arma::mat& parameters,
    const size_t begin,
    arma::mat& gradient,
    const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType, CustomLayers...>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType,
         CustomLayers...>::ResetParameters()
{
  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType,
                        CustomLayers...> networkInit(initializeRule);
  networkInit.Initialize(network, parameter);
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType,
         CustomLayers...>::ResetDeterministic()
{
  DeterministicSetVisitor deterministicSetVisitor(deterministic);
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deterministicSetVisitor));
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType,
         CustomLayers...>::ResetGradients(arma::mat& gradient)
{
  size_t offset = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    offset += boost::apply_visitor(GradientSetVisitor(gradient, offset),
        network[i]);
  }
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename InputType>
void FFNEQL<InitializationRuleType,
         CustomLayers...>::Forward(const InputType& input)
{
  boost::apply_visitor(ForwardVisitor(input,
      boost::apply_visitor(outputParameterVisitor, network.front())),
      network.front());

  if (!reset)
  {
    if (boost::apply_visitor(outputWidthVisitor, network.front()) != 0)
    {
      width = boost::apply_visitor(outputWidthVisitor, network.front());
    }

    if (boost::apply_visitor(outputHeightVisitor, network.front()) != 0)
    {
      height = boost::apply_visitor(outputHeightVisitor, network.front());
    }
  }

  for (size_t i = 1; i < network.size(); ++i)
  {
    if (!reset)
    {
      // Set the input width.
      boost::apply_visitor(SetInputWidthVisitor(width), network[i]);

      // Set the input height.
      boost::apply_visitor(SetInputHeightVisitor(height), network[i]);
    }

    boost::apply_visitor(ForwardVisitor(boost::apply_visitor(
        outputParameterVisitor, network[i - 1]),
        boost::apply_visitor(outputParameterVisitor, network[i])), network[i]);

    if (!reset)
    {
      // Get the output width.
      if (boost::apply_visitor(outputWidthVisitor, network[i]) != 0)
      {
        width = boost::apply_visitor(outputWidthVisitor, network[i]);
      }

      // Get the output height.
      if (boost::apply_visitor(outputHeightVisitor, network[i]) != 0)
      {
        height = boost::apply_visitor(outputHeightVisitor, network[i]);
      }
    }
  }

  if (!reset)
    reset = true;
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType, CustomLayers...>::Backward()
{
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, network.back()), error,
      boost::apply_visitor(deltaVisitor, network.back())), network.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1]),
        boost::apply_visitor(deltaVisitor, network[network.size() - i])),
        network[network.size() - i]);
  }
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename InputType>
void FFNEQL<InitializationRuleType,
         CustomLayers...>::Gradient(const InputType& input)
{
  boost::apply_visitor(GradientVisitor(input,
      boost::apply_visitor(deltaVisitor, network[1])), network.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    boost::apply_visitor(GradientVisitor(boost::apply_visitor(
        outputParameterVisitor, network[i - 1]),
        boost::apply_visitor(deltaVisitor, network[i + 1])), network[i]);
  }

  boost::apply_visitor(GradientVisitor(boost::apply_visitor(
      outputParameterVisitor, network[network.size() - 2]), error),
      network[network.size() - 1]);
}

template<typename typename InitializationRuleType,
         typename... CustomLayers>
template<typename Archive>
void FFNEQL<InitializationRuleType, CustomLayers...>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(parameter));
  ar(CEREAL_NVP(width));
  ar(CEREAL_NVP(height));

  ar(CEREAL_NVP(reset));

  // Be sure to clear other layers before loading.
  if (cereal::is_loading<Archive>())
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
    network.clear();
  }

  ar(CEREAL_VECTOR_VARIANT_POINTER(network));

  // If we are loading, we need to initialize the weights.
  if (cereal::is_loading<Archive>())
  {
    size_t offset = 0;
    for (size_t i = 0; i < network.size(); ++i)
    {
      offset += boost::apply_visitor(WeightSetVisitor(parameter, offset),
          network[i]);

      boost::apply_visitor(resetVisitor, network[i]);
    }

    deterministic = true;
    ResetDeterministic();
  }
}

template<typename InitializationRuleType,
         typename... CustomLayers>
void FFNEQL<InitializationRuleType,
         CustomLayers...>::Swap(FFNEQL& network)
{
  std::swap(initializeRule, network.initializeRule);
  std::swap(width, network.width);
  std::swap(height, network.height);
  std::swap(reset, network.reset);
  std::swap(this->network, network.network);
  std::swap(predictors, network.predictors);
  std::swap(responses, network.responses);
  std::swap(parameter, network.parameter);
  std::swap(numFunctions, network.numFunctions);
  std::swap(error, network.error);
  std::swap(deterministic, network.deterministic);
  std::swap(delta, network.delta);
  std::swap(inputParameter, network.inputParameter);
  std::swap(outputParameter, network.outputParameter);
  std::swap(gradient, network.gradient);
};

template<typename typename InitializationRuleType,
         typename... CustomLayers>
FFNEQL<InitializationRuleType, CustomLayers...>::FFNEQL(
    const FFNEQL& network):
    initializeRule(network.initializeRule),
    width(network.width),
    height(network.height),
    reset(network.reset),
    predictors(network.predictors),
    responses(network.responses),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    error(network.error),
    deterministic(network.deterministic),
    delta(network.delta),
    inputParameter(network.inputParameter),
    outputParameter(network.outputParameter),
    gradient(network.gradient)
{
  // Build new layers according to source network
  for (size_t i = 0; i < network.network.size(); ++i)
  {
    this->network.push_back(boost::apply_visitor(copyVisitor,
        network.network[i]));
    boost::apply_visitor(resetVisitor, this->network.back());
  }
};

template<typename typename InitializationRuleType,
         typename... CustomLayers>
FFNEQL<InitializationRuleType, CustomLayers...>::FFNEQL(
    FFNEQL&& network):
    initializeRule(std::move(network.initializeRule)),
    width(network.width),
    height(network.height),
    reset(network.reset),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    error(std::move(network.error)),
    deterministic(network.deterministic),
    delta(std::move(network.delta)),
    inputParameter(std::move(network.inputParameter)),
    outputParameter(std::move(network.outputParameter)),
    gradient(std::move(network.gradient))
{
  this->network = std::move(network.network);
};

template<typename typename InitializationRuleType,
         typename... CustomLayers>
FFNEQL<InitializationRuleType, CustomLayers...>&
FFNEQL<InitializationRuleType,
    CustomLayers...>::operator = (FFNEQL network)
{
  Swap(network);
  return *this;
};

} // namespace ann
} // namespace mlpack

#endif
