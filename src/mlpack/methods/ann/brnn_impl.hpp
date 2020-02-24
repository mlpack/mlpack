/**
 * @file brnn_impl.hpp
 * @author Saksham Bansal
 *
 * Definition of the BRNN class, which implements bidirectional recurrent
 * neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_BRNN_IMPL_HPP
#define MLPACK_METHODS_ANN_BRNN_IMPL_HPP

// In case it hasn't been included yet.
#include "brnn.hpp"

#include "visitor/load_output_parameter_visitor.hpp"
#include "visitor/save_output_parameter_visitor.hpp"
#include "visitor/forward_visitor.hpp"
#include "visitor/backward_visitor.hpp"
#include "visitor/reset_cell_visitor.hpp"
#include "visitor/gradient_set_visitor.hpp"
#include "visitor/gradient_visitor.hpp"
#include "visitor/weight_set_visitor.hpp"
#include "visitor/run_set_visitor.hpp"

#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::BRNN(
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    MergeLayerType mergeLayer,
    MergeOutputType mergeOutput,
    InitializationRuleType initializeRule) :
    rho(rho),
    outputLayer(std::move(outputLayer)),
    mergeLayer(new MergeLayerType(mergeLayer)),
    mergeOutput(new MergeOutputType(mergeOutput)),
    initializeRule(std::move(initializeRule)),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single),
    numFunctions(0),
    deterministic(true),
    forwardRNN(rho, single, outputLayer, initializeRule),
    backwardRNN(rho, single, outputLayer, initializeRule)
{
  /* Nothing to do here. */
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Train(
    arma::cube predictors,
    arma::cube responses,
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
  }

  if (optimizer.MaxIterations() < this->predictors.n_cols)
  {
    Log::Warn << "The number of iterations is less than the size of the "
              << "dataset, the optimizer might not pass over the entire "
              << "dataset. Please modify the number of iterations to be "
              << "at least equal to the number of columns of your "
              << "dataset." << std::endl;
  }

  // Train the model.
  Timer::Start("BRNN_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("BRNN_optimization");

  Log::Info << "BRNN::BRNN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Train(
    arma::cube predictors,
    arma::cube responses)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
  }

  OptimizerType optimizer;

  if (optimizer.MaxIterations() < this->predictors.n_cols)
  {
    Log::Warn << "The number of iterations is less than the size of the "
              << "dataset, the optimizer might not pass over the entire "
              << "dataset. Please modify the number of iterations to be "
              << "at least equal to the number of columns of your "
              << "dataset." << std::endl;
  }

  // Train the model.
  const double out = optimizer.Optimize(*this, parameter);

  Log::Info << "BRNN::BRNN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Predict(
    arma::cube predictors, arma::cube& results, const size_t batchSize)
{
  forwardRNN.rho = backwardRNN.rho = rho;

  forwardRNN.ResetCells();
  backwardRNN.ResetCells();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (std::is_same<MergeLayerType, Concat<>>::value)
  {
    results = arma::zeros<arma::cube>(outputSize * 2, predictors.n_cols, rho);
  }
  else
  {
    results = arma::zeros<arma::cube>(outputSize, predictors.n_cols, rho);
  }

  std::vector<arma::mat> results1, results2;
  arma::mat input;

  // Forward both RNN's from opposite directions.
  for (size_t begin = 0; begin < predictors.n_cols; begin += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols - begin));
    for (size_t seqNum = 0; seqNum < rho; ++seqNum)
    {
      forwardRNN.Forward(std::move(arma::mat(
          predictors.slice(seqNum).colptr(begin),
          predictors.n_rows, effectiveBatchSize, false, true)));
      backwardRNN.Forward(std::move(arma::mat(
          predictors.slice(rho - seqNum - 1).colptr(begin),
          predictors.n_rows, effectiveBatchSize, false, true)));

      boost::apply_visitor(SaveOutputParameterVisitor(
          std::move(results1)), forwardRNN.network.back());
      boost::apply_visitor(SaveOutputParameterVisitor(
          std::move(results2)), backwardRNN.network.back());
    }
    reverse(results1.begin(), results1.end());

    // Forward outputs from both RNN's through merge layer for each time step.
    for (size_t seqNum = 0; seqNum < rho; ++seqNum)
    {
      boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(results1)), forwardRNN.network.back());
      boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(results2)), backwardRNN.network.back());

      boost::apply_visitor(ForwardVisitor(std::move(input),
          std::move(boost::apply_visitor(outputParameterVisitor, mergeLayer))),
          mergeLayer);
      boost::apply_visitor(ForwardVisitor(
          std::move(boost::apply_visitor(outputParameterVisitor, mergeLayer)),
          std::move(boost::apply_visitor(outputParameterVisitor, mergeOutput))),
          mergeOutput);
      results.slice(seqNum).submat(0, begin, results.n_rows - 1, begin +
          effectiveBatchSize - 1) =
          boost::apply_visitor(outputParameterVisitor, mergeOutput);
    }
  }
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t begin,
    const size_t batchSize,
    const bool deterministic)
{
  forwardRNN.rho = backwardRNN.rho = rho;
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  if (!inputSize)
  {
    inputSize = predictors.n_rows;
    targetSize = responses.n_rows;
  }
  else if (targetSize == 0)
  {
    targetSize = responses.n_rows;
  }

  forwardRNN.ResetCells();
  backwardRNN.ResetCells();

  double performance = 0;
  size_t responseSeq = 0;

  std::vector<arma::mat> results1, results2;
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    forwardRNN.Forward(std::move(arma::mat(
        predictors.slice(seqNum).colptr(begin),
        predictors.n_rows, batchSize, false, true)));
    backwardRNN.Forward(std::move(arma::mat(
        predictors.slice(rho - seqNum - 1).colptr(begin),
        predictors.n_rows, batchSize, false, true)));

    boost::apply_visitor(SaveOutputParameterVisitor(
        std::move(results1)), forwardRNN.network.back());
    boost::apply_visitor(SaveOutputParameterVisitor(
        std::move(results2)), backwardRNN.network.back());
  }
  if (outputSize == 0)
  {
    outputSize = boost::apply_visitor(outputParameterVisitor,
        forwardRNN.network.back()).n_elem / batchSize;
    forwardRNN.outputSize = backwardRNN.outputSize = outputSize;
  }
  reverse(results1.begin(), results1.end());

  // Performance calculation after forwarding through merge layer.
  arma::mat input;
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    if (!single)
    {
      responseSeq = seqNum;
    }
    boost::apply_visitor(LoadOutputParameterVisitor(
        std::move(results1)), forwardRNN.network.back());
    boost::apply_visitor(LoadOutputParameterVisitor(
        std::move(results2)), backwardRNN.network.back());

    boost::apply_visitor(ForwardVisitor(std::move(input),
        std::move(boost::apply_visitor(outputParameterVisitor, mergeLayer))),
        mergeLayer);
    boost::apply_visitor(ForwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, mergeLayer)),
        std::move(boost::apply_visitor(outputParameterVisitor, mergeOutput))),
        mergeOutput);
    performance += outputLayer.Forward(std::move(
        boost::apply_visitor(outputParameterVisitor, mergeOutput)),
        std::move(arma::mat(responses.slice(responseSeq).colptr(begin),
        responses.n_rows, batchSize, false, true)));
  }
  return performance;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Evaluate(
    const arma::mat& parameters,
    const size_t begin,
    const size_t batchSize)
{
  return Evaluate(parameters, begin, batchSize, true);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename GradType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::
EvaluateWithGradient(const arma::mat& /* parameters */,
                     const size_t begin,
                     GradType& gradient,
                     const size_t batchSize)
{
  forwardRNN.rho = backwardRNN.rho = rho;
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      ResetParameters();
    }
    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  if (backwardGradient.is_empty())
  {
    backwardGradient = arma::zeros<arma::mat>(
        parameter.n_rows/ 2,
        parameter.n_cols);
    forwardGradient = arma::zeros<arma::mat>(
        parameter.n_rows/ 2,
        parameter.n_cols);
  }
  if (this->deterministic)
  {
    this->deterministic = false;
    ResetDeterministic();
  }

  if (!inputSize)
  {
    inputSize = predictors.n_rows;
    targetSize = responses.n_rows;
  }
  else if (targetSize == 0)
  {
    targetSize = responses.n_rows;
  }

  forwardRNN.ResetCells();
  backwardRNN.ResetCells();
  size_t networkSize = backwardRNN.network.size();

  // Forward propogation from both directions.
  std::vector<arma::mat> results1, results2;
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    forwardRNN.Forward(std::move(arma::mat(
        predictors.slice(seqNum).colptr(begin),
        predictors.n_rows, batchSize, false, true)));
    backwardRNN.Forward(std::move(arma::mat(
        predictors.slice(rho - seqNum - 1).colptr(begin),
        predictors.n_rows, batchSize, false, true)));

    for (size_t l = 0; l < networkSize; ++l)
    {
      boost::apply_visitor(SaveOutputParameterVisitor(
          std::move(forwardRNNOutputParameter)), forwardRNN.network[l]);
      boost::apply_visitor(SaveOutputParameterVisitor(
          std::move(backwardRNNOutputParameter)), backwardRNN.network[l]);
    }
    boost::apply_visitor(SaveOutputParameterVisitor(
        std::move(results1)), forwardRNN.network.back());
    boost::apply_visitor(SaveOutputParameterVisitor(
        std::move(results2)), backwardRNN.network.back());
  }
  if (outputSize == 0)
  {
    outputSize = boost::apply_visitor(outputParameterVisitor,
        forwardRNN.network.back()).n_elem / batchSize;
    forwardRNN.outputSize = backwardRNN.outputSize = outputSize;
  }

  arma::cube results;
  if (std::is_same<MergeLayerType, Concat<>>::value)
  {
    results = arma::zeros<arma::cube>(outputSize * 2, batchSize, rho);
  }
  else
  {
    results = arma::zeros<arma::cube>(outputSize, batchSize, rho);
  }

  double performance = 0;
  size_t responseSeq = 0;
  arma::mat input;

  reverse(results1.begin(), results1.end());
  // Performance calculation here.
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    if (!single)
    {
      responseSeq = seqNum;
    }
    boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(results1)), forwardRNN.network.back());
    boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(results2)), backwardRNN.network.back());
    boost::apply_visitor(ForwardVisitor(std::move(input),
        std::move(boost::apply_visitor(outputParameterVisitor, mergeLayer))),
        mergeLayer);
    boost::apply_visitor(ForwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, mergeLayer)),
        std::move(results.slice(seqNum))), mergeOutput);
    performance += outputLayer.Forward(std::move(results.slice(seqNum)),
        std::move(arma::mat(responses.slice(responseSeq).colptr(begin),
        responses.n_rows, batchSize, false, true)));
  }

  // Calculate and storing delta parameters from output for t = 1 to T.
  arma::mat delta;
  std::vector<arma::mat> allDelta;

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    if (single && seqNum > 0)
    {
      error.zeros();
    }
    else if (single && seqNum == 0)
    {
      outputLayer.Backward(std::move(results.slice(seqNum)),
          std::move(arma::mat(responses.slice(0).colptr(begin),
          responses.n_rows, batchSize, false, true)), std::move(error));
    }
    else
    {
      outputLayer.Backward(std::move(results.slice(seqNum)),
          std::move(arma::mat(responses.slice(seqNum).colptr(begin),
          responses.n_rows, batchSize, false, true)), std::move(error));
    }

    boost::apply_visitor(BackwardVisitor(std::move(results.slice(seqNum)),
        std::move(error), std::move(delta)), mergeOutput);
    allDelta.push_back(arma::mat(delta));
  }

  // BPTT ForwardRNN from t = T to 1.
  totalGradient = arma::mat(gradient.memptr(),
      parameter.n_elem/2, 1, false, false);

  forwardGradient.zeros();
  forwardRNN.ResetGradients(forwardGradient);
  backwardGradient.zeros();
  backwardRNN.ResetGradients(backwardGradient);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    forwardGradient.zeros();
    for (size_t l = 0; l < networkSize; ++l)
    {
      boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(forwardRNNOutputParameter)),
          forwardRNN.network[networkSize - 1 - l]);
    }
    boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, forwardRNN.network.back())),
        std::move(allDelta[rho - seqNum - 1]), std::move(delta), 0),
        mergeLayer);

    for (size_t i = 2; i < networkSize; ++i)
    {
      boost::apply_visitor(BackwardVisitor(
          std::move(boost::apply_visitor(outputParameterVisitor,
          forwardRNN.network[networkSize - i])),
          std::move(boost::apply_visitor(deltaVisitor,
          forwardRNN.network[networkSize - i + 1])), std::move(
          boost::apply_visitor(deltaVisitor,
          forwardRNN.network[networkSize - i]))),
          forwardRNN.network[networkSize - i]);
    }
    forwardRNN.Gradient(std::move(
        arma::mat(predictors.slice(rho - seqNum - 1).colptr(begin),
        predictors.n_rows, batchSize, false, true)));
    boost::apply_visitor(GradientVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
        forwardRNN.network[networkSize - 2])),
        std::move(allDelta[rho - seqNum - 1]), 0), mergeLayer);
    totalGradient += forwardGradient;
  }

  // BPTT BackwardRNN from t = 1 to T.
  totalGradient = arma::mat(gradient.memptr() + parameter.n_elem/2,
      parameter.n_elem/2, 1, false, false);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    backwardGradient.zeros();
    for (size_t l = 0; l < networkSize; ++l)
    {
      boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(backwardRNNOutputParameter)),
          backwardRNN.network[networkSize - 1 - l]);
    }
    boost::apply_visitor(BackwardVisitor(std::move(
        boost::apply_visitor(outputParameterVisitor,
        backwardRNN.network.back())),
        std::move(allDelta[seqNum]), std::move(delta), 1), mergeLayer);
    for (size_t i = 2; i < networkSize; ++i)
    {
      boost::apply_visitor(BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
        backwardRNN.network[networkSize - i])), std::move(boost::apply_visitor(
        deltaVisitor, backwardRNN.network[networkSize - i + 1])), std::move(
        boost::apply_visitor(deltaVisitor,
        backwardRNN.network[networkSize - i]))),
        backwardRNN.network[networkSize - i]);
    }

    backwardRNN.Gradient(std::move(
        arma::mat(predictors.slice(seqNum).colptr(begin),
        predictors.n_rows, batchSize, false, true)));
    boost::apply_visitor(GradientVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
        backwardRNN.network[networkSize - 2])),
        std::move(allDelta[seqNum]), 1), mergeLayer);
    totalGradient += backwardGradient;
  }
  return performance;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Gradient(
    const arma::mat& parameters,
    const size_t begin,
    arma::mat& gradient,
    const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Shuffle()
{
  arma::cube newPredictors, newResponses;
  math::ShuffleData(predictors, responses, newPredictors, newResponses);

  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
template <class LayerType, class... Args>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Add(Args... args)
{
  forwardRNN.network.push_back(new LayerType(args...));
  backwardRNN.network.push_back(new LayerType(args...));
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::
Add(LayerTypes<CustomLayers...> layer)
{
  forwardRNN.network.push_back(layer);
  backwardRNN.network.push_back(boost::apply_visitor(copyVisitor, layer));
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::ResetParameters()
{
  if (!reset)
  {
    boost::apply_visitor(AddVisitor<CustomLayers...>(
        forwardRNN.network.back()), mergeLayer);
    boost::apply_visitor(AddVisitor<CustomLayers...>(
        backwardRNN.network.back()), mergeLayer);
    boost::apply_visitor(RunSetVisitor(false), mergeLayer);
  }

  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType,
                        CustomLayers...> networkInit(initializeRule);
  size_t rnnWeights = 0;
  for (size_t i = 0; i < forwardRNN.network.size(); ++i)
  {
    rnnWeights += boost::apply_visitor(weightSizeVisitor,
        forwardRNN.network[i]);
  }

  parameter.set_size(2 * rnnWeights, 1);

  forwardRNN.Parameters() = arma::mat(parameter.memptr(),
      rnnWeights, 1, false, false);
  backwardRNN.Parameters() = arma::mat(parameter.memptr() + rnnWeights,
      rnnWeights, 1, false, false);

  // Initialize the forward RNN parameters
  networkInit.Initialize(forwardRNN.network, parameter);

  // Initialize the backward RNN parameters
  networkInit.Initialize(backwardRNN.network, parameter, rnnWeights);

  reset = forwardRNN.reset = backwardRNN.reset = true;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::Reset()
{
  ResetParameters();
  forwardRNN.ResetCells();
  backwardRNN.ResetCells();
  forwardGradient.zeros();
  forwardRNN.ResetGradients(forwardGradient);
  backwardGradient.zeros();
  backwardRNN.ResetGradients(backwardGradient);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::ResetDeterministic()
{
  forwardRNN.deterministic = this->deterministic;
  backwardRNN.deterministic = this->deterministic;
  forwardRNN.ResetDeterministic();
  backwardRNN.ResetDeterministic();
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename Archive>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(backwardRNN);
  ar & BOOST_SERIALIZATION_NVP(forwardRNN);
}

} // namespace ann
} // namespace mlpack

#endif
