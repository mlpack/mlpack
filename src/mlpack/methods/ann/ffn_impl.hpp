/**
 * @file methods/ann/ffn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the FFN class, which implements feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_IMPL_HPP
#define MLPACK_METHODS_ANN_FFN_IMPL_HPP

// In case it hasn't been included yet.
#include "ffn.hpp"

#include "util/gradient_update.hpp"
#include "util/deterministic_update.hpp"
#include "util/loss_update.hpp"
#include "util/output_width_update.hpp"
#include "util/output_height_update.hpp"
#include "util/reset_update.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::FFN(
    OutputLayerType outputLayer, InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    width(0),
    height(0),
    reset(false),
    numFunctions(0),
    deterministic(false)
{
  /* Nothing to do here. */
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::~FFN()
{
  //network.clear();
  for (size_t i = 0; i < network.size(); ++i)
    delete network[i];
}
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    ResetData(InputType predictors, InputType responses)
{
  numFunctions = responses.n_cols;
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
  this->deterministic = false;
  ResetDeterministic();

  if (!reset)
    ResetParameters();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
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

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      !HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
WarnMessageMaxIterations(OptimizerType& /* optimizer */, size_t /* samples */)
    const
{
  // Nothing to do here.
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType, typename... CallbackTypes>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Train(InputType predictors, InputType responses, OptimizerType& optimizer,
        CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameter, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType, typename... CallbackTypes>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Train(InputType predictors, InputType responses,
        CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  OptimizerType optimizer;

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameter, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Forward(const PredictorsType& inputs, ResponsesType& results)
{
  if (parameter.is_empty())
    ResetParameters();

  Forward(inputs);
  results = network.back()->OutputParameter();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Forward(const PredictorsType& inputs, ResponsesType& results,
        const size_t begin, const size_t end)
{
  network[begin]->Forward(inputs, network[begin]->OutputParameter());

  for (size_t i = 1; i < end - begin + 1; ++i)
  {
    network[begin + i]->Forward(network[begin + i - 1]->OutputParameter(),
        network[begin + i]->OutputParameter());
  }

  results = network[end]->OutputParameter();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename TargetsType, typename GradientsType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Backward(const PredictorsType& inputs, const TargetsType& targets,
        GradientsType& gradients)
{
  double res = outputLayer.Forward(network.back()->OutputParameter(), targets);

  for (size_t i = 0; i < network.size(); ++i)
    res += LossUpdate(network[i]);

  outputLayer.Backward(network.back()->OutputParameter(), targets, error);

  gradients = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);

  Backward();
  ResetGradients(gradients);
  Gradient(inputs);

  return res;

}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Predict(InputType predictors, OutputType& results)
{
  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  OutputType resultsTemp;
  Forward(arma::mat(predictors.colptr(0), predictors.n_rows, 1, false, true));

  resultsTemp = network.back()->OutputParameter().col(0);
  results = arma::mat(resultsTemp.n_elem, predictors.n_cols);
  results.col(0) = resultsTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; ++i)
  {
    Forward(arma::mat(predictors.colptr(i), predictors.n_rows, 1, false, true));

    resultsTemp = network.back()->OutputParameter();
    results.col(i) = resultsTemp.col(0);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Evaluate(const PredictorsType& predictors, const ResponsesType& responses)
{
  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  Forward(predictors);

  double res = outputLayer.Forward(network.back()->OutputParameter(),
      responses);
  for (size_t i = 0; i < network.size(); ++i)
    res += LossUpdate(network[i]);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Evaluate(const OutputType& parameters)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += Evaluate(parameters, i, 1, true);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    Evaluate(const OutputType& /* parameters */, const size_t begin,
        const size_t batchSize, const bool deterministic)
{
  if (parameter.is_empty())
    ResetParameters();

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  Forward(predictors.cols(begin, begin + batchSize - 1));

  double res = outputLayer.Forward(network.back()->OutputParameter(),
      responses.cols(begin, begin + batchSize - 1));

  for (size_t i = 0; i < network.size(); ++i)
    res += LossUpdate(network[i]);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Evaluate(const OutputType& parameters, const size_t begin,
    const size_t batchSize)
{
  return Evaluate(parameters, begin, batchSize, true);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
    EvaluateWithGradient(const OutputType& parameters, OutputType& gradient)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += EvaluateWithGradient(parameters, i, gradient, 1);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
EvaluateWithGradient(const OutputType& parameters,
                     const size_t begin,
                     OutputType& gradient,
                     const size_t batchSize)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
      ResetParameters();

    gradient = arma::zeros<OutputType>(parameter.n_rows, parameter.n_cols);
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
  double res = outputLayer.Forward(
      network.back()->OutputParameter(),
      responses.cols(begin, begin + batchSize - 1));

  for (size_t i = 0; i < network.size(); ++i)
    res += network[i]->Loss();

  outputLayer.Backward(network.back()->OutputParameter(),
      responses.cols(begin, begin + batchSize - 1),
      error);

  Backward();
  ResetGradients(gradient);
  Gradient(predictors.cols(begin, begin + batchSize - 1));

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Gradient(const OutputType& parameters, const size_t begin,
    OutputType& gradient, const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
ResetParameters()
{
  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network, parameter);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
ResetDeterministic()
{
  for (size_t i = 0; i < network.size(); ++i)
    DeterministicUpdate(network[i], deterministic);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
ResetGradients(OutputType& gradient)
{
  size_t offset = 0;
  for (size_t i = 0; i < network.size(); ++i)
    offset += GradientUpdate(network[i], gradient, offset);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Forward(const InputType& input)
{
  network.front()->Forward(input, network.front()->OutputParameter());

  if (!reset)
  {
    const size_t outputWidth = OutputWidth(network.front());
    if (outputWidth != 0)
      width = outputWidth;

    const size_t outputHeight = OutputHeight(network.front());
    if (outputHeight != 0)
      height = outputHeight;
  }

  for (size_t i = 1; i < network.size(); ++i)
  {
    if (!reset)
    {
      // Set the input width.
      network[i]->InputWidth() = width;

      // Set the input height.
      network[i]->InputHeight() = height;
    }

    network[i]->Forward(network[i - 1]->OutputParameter(), network[i]->OutputParameter());

    if (!reset)
    {
      // Get the output width.
      const size_t outputWidth = OutputWidth(network[i]);
      if (outputWidth != 0)
        width = outputWidth;

      // Get the output height.
      const size_t outputHeight = OutputHeight(network[i]);
      if (outputHeight != 0)
        height = outputHeight;
    }
  }

  if (!reset)
    reset = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Backward()
{
  network.back()->Backward(network.back()->OutputParameter(),
      error, network.back()->Delta());

  for (size_t i = 2; i < network.size(); ++i)
  {
    network[network.size() - i]->Backward(
        network[network.size() - i]->OutputParameter(),
        network[network.size() - i + 1]->Delta(),
        network[network.size() - i]->Delta());
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Gradient(const InputType& input)
{
  arma::mat inputTemp = input;
  network.front()->Gradient(inputTemp,
      network[1]->Delta(), network.front()->Gradient());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    network[i]->Gradient(network[i - 1]->OutputParameter(),
        network[i + 1]->Delta(),  network[i]->Gradient());
  }

  network[network.size() - 1]->Gradient(network[network.size() - 2]->OutputParameter(),
      error, network[network.size() - 1]->Gradient());
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename Archive>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
serialize(Archive& ar, const uint32_t /* version */)
{
  /* ar(CEREAL_NVP(parameter)); */
  /* ar(CEREAL_NVP(width)); */
  /* ar(CEREAL_NVP(height)); */

  /* ar(CEREAL_NVP(reset)); */

  /* // Be sure to clear other layers before loading. */
  /* if (cereal::is_loading<Archive>()) */
  /* { */
  /*   std::for_each(network.begin(), network.end(), */
  /*       boost::apply_visitor(deleteVisitor)); */
  /*   network.clear(); */
  /* } */

  /* ar(CEREAL_VECTOR_VARIANT_POINTER(network)); */

  /* // If we are loading, we need to initialize the weights. */
  /* if (cereal::is_loading<Archive>()) */
  /* { */
  /*   size_t offset = 0; */
  /*   for (size_t i = 0; i < network.size(); ++i) */
  /*   { */
  /*     offset += boost::apply_visitor(WeightSetVisitor(parameter, offset), */
  /*         network[i]); */

  /*     boost::apply_visitor(resetVisitor, network[i]); */
  /*   } */

  /*   deterministic = true; */
  /*   ResetDeterministic(); */
  /* } */
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
Swap(FFN& network)
{
  std::swap(outputLayer, network.outputLayer);
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

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::FFN(
    const FFN& network):
    outputLayer(network.outputLayer),
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
    this->network.push_back(network.network[i]->Clone());
    ResetUpdate(this->network.back());
  }
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::FFN(
    FFN&& network):
    outputLayer(std::move(network.outputLayer)),
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

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>&
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>::
operator =(FFN network)
{
  Swap(network);
  return *this;
};

} // namespace ann
} // namespace mlpack

#endif
