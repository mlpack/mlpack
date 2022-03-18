/**
 * @file methods/ann/rnn_impl.hpp
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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::RNN(
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    rho(rho),
    single(single),
    network(std::move(outputLayer), std::move(initializeRule))
{
  /* Nothing to do here */
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::RNN(
    const RNN& network) :
    rho(network.rho),
    single(network.single),
    network(network.network)
{
  // Nothing else to do.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::RNN(
    RNN&& network) :
    rho(std::move(network.rho)),
    single(std::move(network.single)),
    network(std::move(network.network))
{
  // Nothing to do here.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::~RNN()
{
  // Nothing special to do.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename OptimizerType, typename... CallbackTypes>
double RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Train(
    arma::Cube<typename InputType::elem_type> predictors,
    arma::Cube<typename OutputType::elem_type> responses,
    OptimizerType& optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  network.WarnMessageMaxIterations(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  network.CheckNetwork("RNN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(*this, network.Parameters(),
      callbacks...);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::Train(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename OptimizerType, typename... CallbackTypes>
double RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Train(
    arma::Cube<typename InputType::elem_type> predictors,
    arma::Cube<typename OutputType::elem_type> responses,
    CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(std::forward(predictors), std::forward(responses), optimizer,
      callbacks...);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Predict(
    arma::Cube<typename InputType::elem_type> predictors,
    arma::Cube<typename OutputType::elem_type>& results,
    const size_t batchSize)
{
  // Ensure that the network is configured correctly.
  network.CheckNetwork("RNN::Predict()", predictors.n_rows, true, false);

  results.set_size(network.network.OutputSize(), predictors.n_cols,
      predictors.n_slices);

  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    Forward(predictors, results, i, effectiveBatchSize);
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Reset(const size_t inputDimensionality)
{
  // This is a reimplementation of FFN::Reset() that correctly prints
  // "RNN::Reset()".
  network.Parameters().clear();

  if (inputDimensionality != 0)
  {
    network.CheckNetwork("RNN::Reset()", inputDimensionality, true, false);
  }
  else
  {
    const size_t inputDims = std::accumulate(network.InputDimensions().begin(),
        network.InputDimensions().end(), 0);
    network.CheckNetwork("RNN::Reset()", inputDims, true, false);
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Forward(
    const arma::Cube<typename InputType::elem_type>& predictors,
    arma::Cube<typename OutputType::elem_type>& results)
{
  Forward(std::forward(predictors), results, 0, predictors.n_cols - 1);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Forward(
    const arma::Cube<typename InputType::elem_type>& predictors,
    arma::Cube<typename OutputType::elem_type>& results,
    const size_t begin,
    const size_t batchSize)
{
  // Ensure the network is valid.
  network.CheckNetwork("RNN::Forward()", predictors.n_rows);

  // This is internal---so we assume the network is already ready to go.
  results.set_size(network.network.OutputSize(), predictors.n_cols,
      predictors.n_slices);

  // Since we aren't doing a backward pass, we don't actually need to store the
  // state for each time step---we can fit it all in one buffer.
  ResetMemoryState(1, batchSize);
  SetPreviousStep(size_t(-1));
  SetCurrentStep(size_t(0));

  // Iterate over all time steps.
  for (size_t t = 0; t < predictors.n_slices; ++t)
  {
    // If it is after the first step, we have a previous state.
    if (t == 1)
      SetPreviousStep(size_t(0));

    // Create aliases for the input and output.
    const arma::Mat<typename InputType::elem_type> inputAlias(
        (typename InputType::elem_type*) predictors.slice(t).colptr(begin),
        predictors.n_rows, batchSize, false, true);
    const size_t responseStep = (single) ? 0 : t;
    arma::Mat<typename OutputType::elem_type> outputAlias(
        responses.slice(responseStep).colptr(begin), responses.n_rows,
        batchSize, false, true);

    network.Forward(inputAlias, outputAlias);
  }

  // TODO: I think we need to store networkOutputs and also all the state!
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename PredictorsType, typename TargetsType, typename GradientsType>
double RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Backward(
    const PredictorsType& inputs,
    const TargetsType& targets,
    GradientsType& gradients)
{
  // Compute the loss for each time step.
  double res = 0.0;
  return res;
  // TODO: finish this
  //for (size_t t = 0; t < inputs.n_slices; ++t)
  //{
    

  //  res += network.outputLayer.Forward(
  //}
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename Archive>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(network));
  ar(CEREAL_NVP(predictors));
  ar(CEREAL_NVP(responses));
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
double RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(
    const OutputType& /* parameters */,
    const size_t begin,
    const size_t batchSize)
{
  // Ensure the network is valid.
  network.CheckNetwork("RNN::Evaluate()", predictors.n_rows);

  // The core of the computation here is to pass through each step.  Since we
  // are not computing the gradient, we can be "clever" and use only one memory
  // cell---we don't need to know about the past.
  ResetMemoryState(1, batchSize);
  SetCurrentStep(0);
  SetPreviousStep(size_t(-1));
  arma::mat output(network.network.OutputSize(), batchSize);

  double loss = 0.0;
  for (size_t t = 0; t < predictors.n_slices; ++t)
  {
    if (t == 1)
      SetPreviousStep(0);

    // Wrap a matrix around our data to avoid a copy.
    arma::mat stepData(predictors.slice(t).colptr(begin), predictors.n_rows,
        batchSize, false, true);
    const size_t responseStep = (single) ? 0 : t;
    arma::mat responseData(responses.slice(responseStep).colptr(begin),
        responses.n_rows, batchSize, false, true);

    // TODO: does this cause a copy?
    network.ResetData(std::move(stepData), std::move(responseData));
    loss += network.Evaluate(output, begin, batchSize);
  }

  return loss;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename GradType>
double RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::EvaluateWithGradient(
    const OutputType& parameters,
    GradType& gradient)
{
  return EvaluateWithGradient(parameters, 0, gradient, predictors.n_cols);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename GradType>
double RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::EvaluateWithGradient(
    const OutputType& /* parameters */,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize)
{
  network.CheckNetwork("RNN::EvaluateWithGradient()", predictors.n_rows);

  double loss = 0;
  // TODO: cleaner
  const size_t effectiveRho = std::min(rho, size_t(responses.n_slices));

  ResetMemoryState(effectiveRho, batchSize);
  SetPreviousStep(size_t(-1));
  arma::Cube<typename OutputType::elem_type> outputs(
      network.network.OutputSize(), batchSize, effectiveRho);

  // If `bpttSteps` is less than the number of time steps in the data, then for
  // the first few steps, we won't actually need to hold onto any historical
  // information, since BPTT will never go back that far.
  const size_t extraSteps = (predictors.n_slices - effectiveRho + 1);
  for (size_t t = 0; t < std::min(size_t(predictors.n_slices), extraSteps); ++t)
  {
    SetCurrentStep(0);

    // Wrap a matrix around our data to avoid a copy.
    arma::mat stepData(predictors.slice(t).colptr(begin), predictors.n_rows,
        batchSize, false, true);
    arma::mat outputData(outputs.slice(t).memptr(), outputs.n_rows,
        outputs.n_cols, false, true);
    network.network.Forward(stepData, outputData);

    const size_t responseStep = (single) ? 0 : t;
    arma::mat responseData(responses.slice(responseStep).colptr(begin),
        responses.n_rows, batchSize, false, true);

    loss += network.outputLayer.Forward(outputData, responseData);

    SetPreviousStep(0);
  }

  // Next, we reach the time steps that will be used for BPTT, for which we must
  // preserve step data.
  for (size_t t = extraSteps; t < predictors.n_slices; ++t)
  {
    SetCurrentStep(t - extraSteps);

    // Wrap a matrix around our data to avoid a copy.
    arma::mat stepData(predictors.slice(t).colptr(begin), predictors.n_rows,
        batchSize, false, true);
    arma::mat outputData(outputs.slice(t).memptr(), outputs.n_rows,
        outputs.n_cols, false, true);
    network.network.Forward(stepData, outputData);

    const size_t responseStep = (single) ? 0 : t;
    arma::mat responseData(responses.slice(responseStep).colptr(begin),
        responses.n_rows, batchSize, false, true);

    loss += network.outputLayer.Forward(outputData, responseData);

    SetPreviousStep(t - extraSteps);
  }

  // Add loss (this is not dependent on time steps, and should only be added
  // once).
  loss += network.network.Loss();

  // Initialize current/working gradient.
  gradient.zeros(network.Parameters().n_rows, network.Parameters().n_cols);
  GradType currentGradient;
  currentGradient.zeros(network.Parameters().n_rows,
      network.Parameters().n_cols);

  SetPreviousStep(size_t(-1));
  for (size_t t = predictors.n_slices - 1; t >= predictors.n_slices - effectiveRho; --t)
  {
    SetCurrentStep(t);

    currentGradient.zeros();
    OutputType error(outputs.n_rows, outputs.n_cols);

    // Set up the response by backpropagating through the output layer.  Note
    // that if we are in 'single' mode, we don't care what the network outputs
    // until the input sequence is done, so there is no error for any timestep
    // other than the first one.
    if (single && t < responses.n_slices - 1)
    {
      error.zeros();
    }
    else
    {
      arma::mat outputData(outputs.slice(t).colptr(0), outputs.n_rows,
          outputs.n_cols, false, true);
      arma::mat respData(responses.slice(t).colptr(begin), responses.n_rows,
          responses.n_cols, false, true);
      network.outputLayer.Backward(outputData, respData, error);
    }

    // Now pass that error backwards through the network.
    arma::mat outputData(outputs.slice(t).colptr(0), outputs.n_rows,
        outputs.n_cols, false, true);
    // TODO: allocate space for networkDelta?
    InputType networkDelta;
    network.network.Backward(outputData, error, networkDelta);

    arma::mat stepData(predictors.slice(t).colptr(begin), predictors.n_rows,
        batchSize, false, true);
    network.network.Gradient(stepData, error, currentGradient);
    gradient += currentGradient;

    SetPreviousStep(t);
  }

  return loss;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
template<typename GradType>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Gradient(
    const OutputType& parameters,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Shuffle()
{
  arma::Cube<typename InputType::elem_type> newPredictors;
  arma::Cube<typename OutputType::elem_type> newResponses;
  math::ShuffleData(predictors, responses, newPredictors, newResponses);

  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::ResetData(
    arma::Cube<typename InputType::elem_type> predictors,
    arma::Cube<typename OutputType::elem_type> responses)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::ResetMemoryState(const size_t memorySize, const size_t batchSize)
{
  // Iterate over all layers and set the memory size.
  for (Layer<InputType, OutputType>* l : network.Network())
  {
    // We can only call ClearRecurrentState() on RecurrentLayers.
    RecurrentLayer<InputType, OutputType>* r =
        dynamic_cast<RecurrentLayer<InputType, OutputType>*>(l);
    if (r != nullptr)
      r->ClearRecurrentState(memorySize, batchSize);
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::SetPreviousStep(const size_t step)
{
  // Iterate over all layers and set the memory size.
  for (Layer<InputType, OutputType>* l : network.Network())
  {
    // We can only call SetPreviousStep() on RecurrentLayers.
    RecurrentLayer<InputType, OutputType>* r =
        dynamic_cast<RecurrentLayer<InputType, OutputType>*>(l);
    if (r != nullptr)
      r->PreviousStep() = step;
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename InputType,
    typename OutputType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::SetCurrentStep(const size_t step)
{
  // Iterate over all layers and set the memory size.
  for (Layer<InputType, OutputType>* l : network.Network())
  {
    // We can only call SetPreviousStep() on RecurrentLayers.
    RecurrentLayer<InputType, OutputType>* r =
        dynamic_cast<RecurrentLayer<InputType, OutputType>*>(l);
    if (r != nullptr)
      r->CurrentStep() = step;
  }
}

} // namespace ann
} // namespace mlpack

#endif
