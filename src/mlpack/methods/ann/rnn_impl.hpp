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
#include "layer/recurrent_layer.hpp"

namespace mlpack {

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::RNN(
    const size_t bpttSteps,
    const bool single,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    bpttSteps(bpttSteps),
    single(single),
    network(std::move(outputLayer), std::move(initializeRule))
{
  /* Nothing to do here */
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::RNN(
    const RNN& network) :
    bpttSteps(network.bpttSteps),
    single(network.single),
    network(network.network)
{
  // Nothing else to do.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::RNN(
    RNN&& network) :
    bpttSteps(std::move(network.bpttSteps)),
    single(std::move(network.single)),
    network(std::move(network.network))
{
  // Nothing to do here.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>&
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::operator=(const RNN& other)
{
  if (this != &other)
  {
    bpttSteps = other.bpttSteps;
    single = other.single;
    network = other.network;
    predictors.clear();
    responses.clear();
  }

  return *this;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>&
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::operator=(RNN&& other)
{
  if (this != &other)
  {
    bpttSteps = std::move(other.bpttSteps);
    single = std::move(other.single);
    network = std::move(other.network);
    predictors.clear();
    responses.clear();
  }

  return *this;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::~RNN()
{
  // Nothing special to do.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(
    arma::Cube<typename MatType::elem_type> predictors,
    arma::Cube<typename MatType::elem_type> responses,
    OptimizerType& optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  network.WarnMessageMaxIterations(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  network.CheckNetwork("RNN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("rnn_optimization");
  const typename MatType::elem_type out =
      optimizer.Optimize(*this, network.Parameters(), callbacks...);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::Train(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(
    arma::Cube<typename MatType::elem_type> predictors,
    arma::Cube<typename MatType::elem_type> responses,
    CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(std::move(predictors), std::move(responses), optimizer,
      callbacks...);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Predict(
    const arma::Cube<typename MatType::elem_type>& predictors,
    arma::Cube<typename MatType::elem_type>& results,
    const size_t batchSize)
{
  // Ensure that the network is configured correctly.
  network.CheckNetwork("RNN::Predict()", predictors.n_rows, true, false);

  results.set_size(network.network.OutputSize(), predictors.n_cols,
      predictors.n_slices);

  MatType inputAlias, outputAlias;
  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    // Since we aren't doing a backward pass, we don't actually need to store
    // the state for each time step---we can fit it all in one buffer.
    ResetMemoryState(1, effectiveBatchSize);
    SetPreviousStep(size_t(-1));
    SetCurrentStep(size_t(0));

    // Iterate over all time steps.
    for (size_t t = 0; t < predictors.n_slices; ++t)
    {
      // If it is after the first step, we have a previous state.
      if (t == 1)
        SetPreviousStep(size_t(0));

      // Create aliases for the input and output.
      MakeAlias(inputAlias, predictors.slice(t), predictors.n_rows,
          effectiveBatchSize, i * predictors.slice(t).n_rows);
      MakeAlias(outputAlias, results.slice(t), results.n_rows,
          effectiveBatchSize, i * results.slice(t).n_rows);

      network.Forward(inputAlias, outputAlias);
    }
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
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
    typename MatType
>
template<typename Archive>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  #ifndef MLPACK_ENABLE_ANN_SERIALIZATION
    // Note: if you define MLPACK_IGNORE_ANN_SERIALIZATION_WARNING, you had
    // better ensure that every layer you are serializing has had
    // CEREAL_REGISTER_TYPE() called somewhere.  See layer/serialization.hpp for
    // more information.
    #ifndef MLPACK_IGNORE_ANN_SERIALIZATION_WARNING
      throw std::runtime_error("Cannot serialize a neural network unless "
          "MLPACK_ENABLE_ANN_SERIALIZATION is defined!  See the \"Additional "
          "build options\" section of the README for more information.");
    #endif
  #else
    ar(CEREAL_NVP(bpttSteps));
    ar(CEREAL_NVP(single));
    ar(CEREAL_NVP(network));

    if (Archive::is_loading::value)
    {
      // We can clear these members, since it's not possible to serialize in the
      // middle of training and resume.
      predictors.clear();
      responses.clear();
    }
  #endif
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
typename MatType::elem_type RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(
    const MatType& /* parameters */,
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
  MatType output(network.network.OutputSize(), batchSize);

  typename MatType::elem_type loss = 0.0;
  MatType stepData, responseData;
  for (size_t t = 0; t < predictors.n_slices; ++t)
  {
    if (t == 1)
      SetPreviousStep(0);

    // Manually reset the data of the network to be an alias of the current time
    // step.
    MakeAlias(network.predictors, predictors.slice(t), predictors.n_rows,
        batchSize, begin * predictors.slice(t).n_rows);
    const size_t responseStep = (single) ? 0 : t;
    MakeAlias(network.responses, responses.slice(responseStep),
        responses.n_rows, batchSize,
        begin * responses.slice(responseStep).n_rows);

    loss += network.Evaluate(output, begin, batchSize);
  }

  return loss;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
template<typename GradType>
typename MatType::elem_type RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::EvaluateWithGradient(
    const MatType& parameters,
    GradType& gradient)
{
  return EvaluateWithGradient(parameters, 0, gradient, predictors.n_cols);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
template<typename GradType>
typename MatType::elem_type RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::EvaluateWithGradient(
    const MatType& /* parameters */,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize)
{
  network.CheckNetwork("RNN::EvaluateWithGradient()", predictors.n_rows);

  typename MatType::elem_type loss = 0;

  // We must save anywhere between 1 and `bpttSteps` states, but we are limited
  // by `predictors.n_slices`.
  const size_t effectiveBPTTSteps = std::max(size_t(1),
      std::min(bpttSteps, size_t(predictors.n_slices)));

  ResetMemoryState(effectiveBPTTSteps, batchSize);
  SetPreviousStep(size_t(-1));
  arma::Cube<typename MatType::elem_type> outputs(
      network.network.OutputSize(), batchSize, effectiveBPTTSteps);

  // If `bpttSteps` is less than the number of time steps in the data, then for
  // the first few steps, we won't actually need to hold onto any historical
  // information, since BPTT will never go back that far.
  const size_t extraSteps = (predictors.n_slices - effectiveBPTTSteps + 1);
  MatType stepData, outputData, responseData;
  for (size_t t = 0; t < std::min(size_t(predictors.n_slices), extraSteps); ++t)
  {
    SetCurrentStep(0);

    // Make an alias of the step's data.
    MakeAlias(stepData, predictors.slice(t), predictors.n_rows, batchSize,
        begin * predictors.slice(t).n_rows);
    MakeAlias(outputData, outputs.slice(t), outputs.n_rows, outputs.n_cols);
    network.network.Forward(stepData, outputData);

    const size_t responseStep = (single) ? 0 : t;
    MakeAlias(responseData, responses.slice(responseStep),
        responses.n_rows, batchSize,
        begin * responses.slice(responseStep).n_rows);

    loss += network.outputLayer.Forward(outputData, responseData);

    SetPreviousStep(0);
  }

  // Next, we reach the time steps that will be used for BPTT, for which we must
  // preserve step data.
  for (size_t t = extraSteps; t < predictors.n_slices; ++t)
  {
    SetCurrentStep(t - extraSteps + 1);

    // Wrap a matrix around our data to avoid a copy.
    MakeAlias(stepData, predictors.slice(t), predictors.n_rows, batchSize,
        begin * predictors.slice(t).n_rows);
    MakeAlias(outputData, outputs.slice(t), outputs.n_rows, outputs.n_cols);
    network.network.Forward(stepData, outputData);

    const size_t responseStep = (single) ? 0 : t;
    MakeAlias(responseData, responses.slice(responseStep),
        responses.n_rows, batchSize,
        begin * responses.slice(responseStep).n_rows);

    loss += network.outputLayer.Forward(outputData, responseData);

    SetPreviousStep(t - extraSteps + 1);
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
  const size_t minStep = predictors.n_slices - effectiveBPTTSteps + 1;
  for (size_t t = predictors.n_slices; t >= minStep; --t)
  {
    SetCurrentStep(t - 1);

    currentGradient.zeros();
    MatType error(outputs.n_rows, outputs.n_cols);

    // Set up the response by backpropagating through the output layer.  Note
    // that if we are in 'single' mode, we don't care what the network outputs
    // until the input sequence is done, so there is no error for any timestep
    // other than the first one.
    if (single && (t - 1) < responses.n_slices - 1)
    {
      error.zeros();
    }
    else
    {
      MakeAlias(outputData, outputs.slice(t - 1), outputs.n_rows,
          outputs.n_cols);
      const size_t respStep = (single) ? 0 : t - 1;
      MakeAlias(responseData, responses.slice(respStep), responses.n_rows,
          batchSize, begin * responses.slice(respStep).n_rows);
      network.outputLayer.Backward(outputData, responseData, error);
    }

    // Now pass that error backwards through the network.
    MakeAlias(stepData, predictors.slice(t - 1), predictors.n_rows, batchSize,
        begin * predictors.slice(t - 1).n_rows);
    MakeAlias(outputData, outputs.slice(t - 1), outputs.n_rows,
        outputs.n_cols);

    MatType networkDelta;
    network.network.Backward(stepData, outputData, error, networkDelta);

    network.network.Gradient(stepData, error, currentGradient);
    gradient += currentGradient;

    SetPreviousStep(t - 1);
  }

  return loss;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
template<typename GradType>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Gradient(
    const MatType& parameters,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Shuffle()
{
  ShuffleData(predictors, responses, predictors, responses);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::ResetData(
    arma::Cube<typename MatType::elem_type> predictors,
    arma::Cube<typename MatType::elem_type> responses)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::ResetMemoryState(const size_t memorySize, const size_t batchSize)
{
  // Iterate over all layers and set the memory size.
  for (Layer<MatType>* l : network.Network())
  {
    // We can only call ClearRecurrentState() on RecurrentLayers.
    RecurrentLayer<MatType>* r =
        dynamic_cast<RecurrentLayer<MatType>*>(l);
    if (r != nullptr)
      r->ClearRecurrentState(memorySize, batchSize);
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetPreviousStep(const size_t step)
{
  // Iterate over all layers and set the memory size.
  for (Layer<MatType>* l : network.Network())
  {
    // We can only call SetPreviousStep() on RecurrentLayers.
    RecurrentLayer<MatType>* r =
        dynamic_cast<RecurrentLayer<MatType>*>(l);
    if (r != nullptr)
      r->PreviousStep() = step;
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    typename MatType
>
void RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetCurrentStep(const size_t step)
{
  // Iterate over all layers and set the memory size.
  for (Layer<MatType>* l : network.Network())
  {
    // We can only call SetPreviousStep() on RecurrentLayers.
    RecurrentLayer<MatType>* r =
        dynamic_cast<RecurrentLayer<MatType>*>(l);
    if (r != nullptr)
      r->CurrentStep() = step;
  }
}

} // namespace mlpack

#endif
