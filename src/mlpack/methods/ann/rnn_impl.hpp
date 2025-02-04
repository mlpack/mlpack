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
  ResetData(std::move(predictors), std::move(responses), arma::urowvec());

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
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type RNN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(
    arma::Cube<typename MatType::elem_type> predictors,
    arma::Cube<typename MatType::elem_type> responses,
    arma::urowvec sequenceLengths,
    OptimizerType& optimizer,
    CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses),
      std::move(sequenceLengths));

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
    arma::urowvec sequenceLengths,
    CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(std::move(predictors), std::move(responses),
      std::move(sequenceLengths), optimizer, callbacks...);
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
      single ? 1 : predictors.n_slices);

  MatType inputAlias, outputAlias;
  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    // Since we aren't doing a backward pass, we don't actually need to store
    // the state for each time step---we can fit it all in one buffer.
    ResetMemoryState(0, effectiveBatchSize);

    // Iterate over all time steps.
    for (size_t t = 0; t < predictors.n_slices; ++t)
    {
      SetCurrentStep(t, (t == predictors.n_slices - 1));

      // Create aliases for the input and output.  If we are in single mode, we
      // always output into the same slice.
      MakeAlias(inputAlias, predictors.slice(t), predictors.n_rows,
          effectiveBatchSize, i * predictors.n_rows);
      MakeAlias(outputAlias, results.slice(single ? 0 : t), results.n_rows,
          effectiveBatchSize, i * results.n_rows);

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
>::Predict(
    const arma::Cube<typename MatType::elem_type>& predictors,
    arma::Cube<typename MatType::elem_type>& results,
    const arma::urowvec& sequenceLengths)
{
  // Ensure that the network is configured correctly.
  network.CheckNetwork("RNN::Predict()", predictors.n_rows, true, false);

  results.set_size(network.network.OutputSize(), predictors.n_cols,
      single ? 1 : predictors.n_slices);

  MatType inputAlias, outputAlias;
  for (size_t i = 0; i < predictors.n_cols; i++)
  {
    // Since we aren't doing a backward pass, we don't actually need to store
    // the state for each time step---we can fit it all in one buffer.
    ResetMemoryState(0, 1);

    // Iterate over all time steps.
    const size_t steps = sequenceLengths[i];
    for (size_t t = 0; t < steps; ++t)
    {
      SetCurrentStep(t, (t == steps - 1));

      // Create aliases for the input and output.  If we are in single mode, we
      // always output into the same slice.
      MakeAlias(inputAlias, predictors.slice(t), predictors.n_rows, 1,
          i * predictors.n_rows);
      MakeAlias(outputAlias, results.slice(single ? 0 : t), results.n_rows, 1,
          i * results.n_rows);

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
>::serialize(Archive& ar, const uint32_t /* version */)
{
  #if !defined(MLPACK_ENABLE_ANN_SERIALIZATION) && \
      !defined(MLPACK_ANN_IGNORE_SERIALIZATION_WARNING)
    // Note: if you define MLPACK_IGNORE_ANN_SERIALIZATION_WARNING, you had
    // better ensure that every layer you are serializing has had
    // CEREAL_REGISTER_TYPE() called somewhere.  See layer/serialization.hpp for
    // more information.
    throw std::runtime_error("Cannot serialize a neural network unless "
        "MLPACK_ENABLE_ANN_SERIALIZATION is defined!  See the \"Additional "
        "build options\" section of the README for more information.");

    (void) ar;
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
      sequenceLengths.clear();
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
  MatType output(network.network.OutputSize(), batchSize);

  if (sequenceLengths.n_elem > 0 && batchSize != 1)
    throw std::invalid_argument("Batch size must be 1 for ragged sequences!");

  typename MatType::elem_type loss = 0.0;
  MatType stepData, responseData;
  const size_t steps = (sequenceLengths.n_elem == 0) ? predictors.n_slices :
      sequenceLengths[begin];
  for (size_t t = 0; t < steps; ++t)
  {
    // Manually reset the data of the network to be an alias of the current time
    // step.
    SetCurrentStep(t, (t == steps));
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

  if (sequenceLengths.n_elem > 0 && batchSize != 1)
    throw std::invalid_argument("Batch size must be 1 for ragged sequences!");

  typename MatType::elem_type loss = 0;

  // We must save anywhere between 1 and `bpttSteps` states, but we are limited
  // by `predictors.n_slices`.
  const size_t effectiveBPTTSteps = std::max(size_t(1),
      std::min(bpttSteps, size_t(predictors.n_slices)));

  ResetMemoryState(effectiveBPTTSteps, batchSize);

  // This will store the outputs of the network at each time step.  Note that we
  // only need to store `effectiveBPTTSteps` of output.  We will treat `outputs`
  // as a circular buffer.
  arma::Cube<typename MatType::elem_type> outputs(
      network.network.OutputSize(), batchSize, effectiveBPTTSteps);

  MatType stepData, outputData, responseData;

  // Initialize gradient.
  gradient.zeros(network.Parameters().n_rows, network.Parameters().n_cols);

  // Add loss (this is not dependent on time steps, and should only be added
  // once).  This is, e.g., regularizer loss, and other additive losses not
  // having to do with the output layer.
  loss += network.network.Loss();

  // For backpropagation through time, we must backpropagate for every
  // subsequence of length `bpttSteps`.  Before we've taken `bpttSteps` though,
  // we will be backpropagating shorter sequences.
  const size_t steps = (sequenceLengths.n_elem == 0) ? predictors.n_slices :
      sequenceLengths[begin];
  for (size_t t = 0; t < steps; ++t)
  {
    SetCurrentStep(t, (t == (steps - 1)));

    // Make an alias of the step's data for the forward pass.
    MakeAlias(stepData, predictors.slice(t), predictors.n_rows, batchSize,
        begin * predictors.slice(t).n_rows);
    MakeAlias(outputData, outputs.slice(t % effectiveBPTTSteps), outputs.n_rows,
        outputs.n_cols);
    network.network.Forward(stepData, outputData);

    // Determine what the response should be.  If we are in single mode but not
    // at the end of the sequence, we don't do a backwards pass.
    if (single && t != steps - 1)
    {
      continue;
    }

    // Now backpropagate through time, starting with the current time step and
    // moving backwards.
    MatType error;
    for (size_t step = 0; step < std::min(t + 1, effectiveBPTTSteps); ++step)
    {
      SetCurrentStep(t - step, (step == 0));

      if (step > 0)
      {
        // Past the first step, the error is zero; only recurrent terms matter.
        error.zeros();

        MakeAlias(stepData, predictors.slice(t - step), predictors.n_rows,
            batchSize, begin * predictors.slice(t - step).n_rows);
        MakeAlias(outputData, outputs.slice((t - step) % effectiveBPTTSteps),
            outputs.n_rows, outputs.n_cols);
      }
      else
      {
        // Otherwise, use the backward pass on the output layer to compute the
        // error.
        const size_t responseStep = (single) ? 0 : t - step;
        MakeAlias(stepData, predictors.slice(t - step), predictors.n_rows,
            batchSize, begin * predictors.slice(t - step).n_rows);
        MakeAlias(responseData, responses.slice(responseStep), responses.n_rows,
            batchSize, begin * responses.slice(responseStep).n_rows);
        MakeAlias(outputData, outputs.slice((t - step) % effectiveBPTTSteps),
            outputs.n_rows, outputs.n_cols);

        // We only need to do this on the first time step of BPTT.
        loss += network.outputLayer.Forward(outputData, responseData);

        // Compute the output error.
        network.outputLayer.Backward(outputData, responseData, error);
      }

      // Now backpropagate that error through the network, and compute the
      // gradient.
      //
      // TODO: note that we could avoid the copy of currentGradient by having
      // each layer *add* its gradient to `gradient`.  However that would
      // require some amount of refactoring.
      MatType networkDelta;
      GradType currentGradient(gradient.n_rows, gradient.n_cols,
          GetFillType<MatType>::zeros);
      network.network.Backward(stepData, outputData, error, networkDelta);
      network.network.Gradient(stepData, error, currentGradient);

      gradient += currentGradient;
    }
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
    arma::Cube<typename MatType::elem_type> responses,
    arma::urowvec sequenceLengths)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
  this->sequenceLengths = std::move(sequenceLengths);
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
>::SetCurrentStep(const size_t step, const bool end)
{
  // Iterate over all layers and set the memory size.
  for (Layer<MatType>* l : network.Network())
  {
    // We can only call CurrentStep() on RecurrentLayers.
    RecurrentLayer<MatType>* r =
        dynamic_cast<RecurrentLayer<MatType>*>(l);
    if (r != nullptr)
      r->CurrentStep(step, end);
  }
}

} // namespace mlpack

#endif
