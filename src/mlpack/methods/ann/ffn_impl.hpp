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

namespace mlpack {

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::FFN(OutputLayerType outputLayer, InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false)
{
  /* Nothing to do here. */
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::FFN(const FFN& network):
    outputLayer(network.outputLayer),
    initializeRule(network.initializeRule),
    network(network.network),
    parameters(network.parameters),
    inputDimensions(network.inputDimensions),
    predictors(network.predictors),
    responses(network.responses),
    // These will be set correctly in the first Forward() call.
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false)
{
  // Nothing to do.
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::FFN(FFN&& network):
    outputLayer(std::move(network.outputLayer)),
    initializeRule(std::move(network.initializeRule)),
    network(std::move(network.network)),
    parameters(std::move(network.parameters)),
    inputDimensions(std::move(network.inputDimensions)),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    // Aliases will not be correct after a std::move(), so we will manually
    // reset them.
    layerMemoryIsSet(false),
    inputDimensionsAreSet(std::move(network.inputDimensionsAreSet))
{
  // Nothing to do.
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
FFN<OutputLayerType, InitializationRuleType, MatType>& FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::operator=(const FFN& other)
{
  if (this != &other)
  {
    outputLayer = other.outputLayer;
    initializeRule = other.initializeRule;
    network = other.network;
    parameters = other.parameters;
    inputDimensions = other.inputDimensions;
    predictors = other.predictors;
    responses = other.responses;
    networkOutput = other.networkOutput;
    networkDelta = other.networkDelta;
    error = other.error;
    inputDimensionsAreSet = other.inputDimensionsAreSet;

    // Copying will not preserve Armadillo aliases correctly, so we will reset
    // those.
    layerMemoryIsSet = false;
  }

  return *this;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
FFN<OutputLayerType, InitializationRuleType, MatType>& FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::operator=(FFN&& other)
{
  if (this != &other)
  {
    outputLayer = std::move(other.outputLayer);
    initializeRule = std::move(other.initializeRule);
    network = std::move(other.network);
    parameters = std::move(other.parameters);
    inputDimensions = std::move(other.inputDimensions);
    predictors = std::move(other.predictors);
    responses = std::move(other.responses);
    networkOutput = std::move(other.networkOutput);
    networkDelta = std::move(other.networkDelta);
    error = std::move(other.error);
    inputDimensionsAreSet = std::move(other.inputDimensionsAreSet);
    layerMemoryIsSet = std::move(other.layerMemoryIsSet);
  }

  return *this;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(MatType predictors,
         MatType responses,
         OptimizerType& optimizer,
         CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  CheckNetwork("FFN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("ffn_optimization");
  const typename MatType::elem_type out =
      optimizer.Optimize(*this, parameters, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::Train(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(MatType predictors,
         MatType responses,
         CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(std::move(predictors), std::move(responses), optimizer,
      callbacks...);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Predict(const MatType& predictors, MatType& results, const size_t batchSize)
{
  // Ensure that the network is configured correctly.
  CheckNetwork("FFN::Predict()", predictors.n_rows, true, false);

  results.set_size(network.OutputSize(), predictors.n_cols);

  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    MatType predictorAlias, resultAlias;

    MakeAlias(predictorAlias, predictors, predictors.n_rows,
        effectiveBatchSize, i * predictors.n_rows);
    MakeAlias(resultAlias, results, results.n_rows, effectiveBatchSize,
        i * results.n_rows);

    network.Forward(predictorAlias, resultAlias);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
size_t FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WeightSize()
{
  // If the input dimensions have not yet been propagated to the network, we
  // must do that now.
  UpdateDimensions("FFN::WeightSize()");
  return network.WeightSize();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Reset(const size_t inputDimensionality)
{
  parameters.clear();

  // If the user provided an input dimensionality, then we will take that as the
  // new input size.  Otherwise, if anything is currently specified in
  // `InputDimensions()`, that will be used.  If nothing is specified in
  // `InputDimensions()`, an exception will be thrown.
  if (inputDimensionality != 0)
  {
    CheckNetwork("FFN::Reset()", inputDimensionality, true, false);
  }
  else if (inputDimensions.size() > 0)
  {
    size_t inputDim = inputDimensions[0];
    for (size_t i = 1; i < inputDimensions.size(); i++)
      inputDim *= inputDimensions[i];
    CheckNetwork("FFN::Reset()", inputDim, true, false);
  }
  else
  {
    throw std::invalid_argument("FFN::Reset(): cannot reset network when no "
        "input dimensionality is given, and `InputDimensions()` has not been "
        "set!");
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetNetworkMode(const bool training)
{
  network.Training() = training;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Forward(const MatType& inputs, MatType& results)
{
  Forward(inputs, results, 0, network.Network().size() - 1);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Forward(const MatType& inputs,
           MatType& results,
           const size_t begin,
           const size_t end)
{
  // Sanity checking...
  if (end < begin)
    return;

  // Ensure the network is valid.
  CheckNetwork("FFN::Forward()", inputs.n_rows);

  // We must always store a copy of the forward pass in `networkOutputs` in case
  // we do a backward pass.
  networkOutput.set_size(network.OutputSize(), inputs.n_cols);
  network.Forward(inputs, networkOutput, begin, end);

  // It's possible the user passed `networkOutput` as `results`; in this case,
  // we don't need to create an alias.
  if (&results != &networkOutput)
    results = networkOutput;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Backward(const MatType& inputs,
            const MatType& targets,
            MatType& gradients)
{
  const typename MatType::elem_type res =
      outputLayer.Forward(networkOutput, targets) + network.Loss();

  // Compute the error of the output layer.
  outputLayer.Backward(networkOutput, targets, error);

  // Perform the backward pass.
  network.Backward(inputs, networkOutput, error, networkDelta);

  // Now compute the gradients.
  // The gradient should have the same size as the parameters.
  gradients.set_size(parameters.n_rows, parameters.n_cols);
  network.Gradient(inputs, error, gradients);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(const MatType& predictors, const MatType& responses)
{
  // Sanity check: ensure network is valid.
  CheckNetwork("FFN::Evaluate()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  network.Forward(predictors, networkOutput);

  return outputLayer.Forward(networkOutput, responses) + network.Loss();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename Archive>
void FFN<
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
    // Serialize the output layer and initialization rule.
    ar(CEREAL_NVP(outputLayer));
    ar(CEREAL_NVP(initializeRule));

    // Serialize the network itself.
    ar(CEREAL_NVP(network));
    ar(CEREAL_NVP(parameters));

    // Serialize the expected input size.
    ar(CEREAL_NVP(inputDimensions));

    // If we are loading, we need to initialize the weights.
    if (cereal::is_loading<Archive>())
    {
      // We can clear these members, since it's not possible to serialize in the
      // middle of training and resume.
      predictors.clear();
      responses.clear();

      networkOutput.clear();
      networkDelta.clear();

      layerMemoryIsSet = false;
      inputDimensionsAreSet = false;

      // The weights in `parameters` will be correctly set for each layer in the
      // first call to Forward().
    }
  #endif
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(const MatType& parameters)
{
  typename MatType::elem_type res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += Evaluate(parameters, i, 1);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(const MatType& /* parameters */,
            const size_t begin,
            const size_t batchSize)
{
  CheckNetwork("FFN::Evaluate()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  networkOutput.set_size(network.OutputSize(), batchSize);
  MatType predictorsBatch, responsesBatch;
  MakeAlias(predictorsBatch, predictors, predictors.n_rows, batchSize,
      begin * predictors.n_rows);
  MakeAlias(responsesBatch, responses, responses.n_rows, batchSize,
      begin * responses.n_rows);
  network.Forward(predictorsBatch, networkOutput);

  return outputLayer.Forward(networkOutput, responsesBatch) + network.Loss();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::EvaluateWithGradient(const MatType& parameters, MatType& gradient)
{
  typename MatType::elem_type res = 0;
  res += EvaluateWithGradient(parameters, 0, gradient, 1);
  MatType tmpGradient(gradient.n_rows, gradient.n_cols,
      GetFillType<MatType>::none);
  for (size_t i = 1; i < predictors.n_cols; ++i)
  {
    res += EvaluateWithGradient(parameters, i, tmpGradient, 1);
    gradient += tmpGradient;
  }

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::EvaluateWithGradient(const MatType& parameters,
                        const size_t begin,
                        MatType& gradient,
                        const size_t batchSize)
{
  CheckNetwork("FFN::EvaluateWithGradient()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  networkOutput.set_size(network.OutputSize(), batchSize);

  // Alias the batches so we don't copy memory.
  MatType predictorsBatch, responsesBatch;
  MakeAlias(predictorsBatch, predictors, predictors.n_rows,
      batchSize, begin * predictors.n_rows);
  MakeAlias(responsesBatch, responses, responses.n_rows,
      batchSize, begin * responses.n_rows);

  network.Forward(predictorsBatch, networkOutput);

  const typename MatType::elem_type obj = outputLayer.Forward(networkOutput,
      responsesBatch) + network.Loss();

  // Now perform the backward pass.
  outputLayer.Backward(networkOutput, responsesBatch, error);

  // The delta should have the same size as the input.
  networkDelta.set_size(predictors.n_rows, batchSize);
  network.Backward(predictorsBatch, networkOutput, error, networkDelta);

  // Now compute the gradients.
  // The gradient should have the same size as the parameters.
  gradient.set_size(parameters.n_rows, parameters.n_cols);
  network.Gradient(predictorsBatch, error, gradient);

  return obj;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Gradient(const MatType& parameters,
            const size_t begin,
            MatType& gradient,
            const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Shuffle()
{
  ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::ResetData(MatType predictors, MatType responses)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  // Set the network to training mode.
  SetNetworkMode(true);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::InitializeWeights()
{
  // Reset the network parameters with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network.Network(), parameters);
  // Override the weight matrix if necessary.
  network.CustomInitialize(parameters, network.WeightSize());
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetLayerMemory()
{
  size_t totalWeightSize = network.WeightSize();

  Log::Assert(totalWeightSize == parameters.n_elem,
      "FFN::SetLayerMemory(): total layer weight size does not match parameter "
      "size!");

  network.SetWeights(parameters);
  layerMemoryIsSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::CheckNetwork(const std::string& functionName,
                const size_t inputDimensionality,
                const bool setMode,
                const bool training)
{
  // If the network is empty, we can't do anything.
  if (network.Network().size() == 0)
  {
    throw std::invalid_argument(functionName + ": cannot use network with no "
        "layers!");
  }

  // Next, check that the input dimensions for each layer are correct.  Note
  // that this will throw an exception if the user has passed data that does not
  // match this->inputDimensions.
  if (!inputDimensionsAreSet)
    UpdateDimensions(functionName, inputDimensionality);

  // We may need to initialize the `parameters` matrix if it is empty or the
  // wrong size.
  if (parameters.is_empty())
  {
    InitializeWeights();
  }
  else if (parameters.n_elem != network.WeightSize())
  {
    parameters.clear();
    InitializeWeights();
  }

  // Make sure each layer is pointing at the right memory.
  if (!layerMemoryIsSet)
    SetLayerMemory();

  // Finally, set the layers of the network to the right mode if the user
  // requested it.
  if (setMode)
    SetNetworkMode(training);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::UpdateDimensions(const std::string& functionName,
                    const size_t inputDimensionality)
{
  // If the input dimensions are completely unset, then assume our input is
  // flat.
  if (inputDimensions.size() == 0)
    inputDimensions = { inputDimensionality };

  size_t totalInputSize = 1;
  for (size_t i = 0; i < inputDimensions.size(); ++i)
    totalInputSize *= inputDimensions[i];

  if (totalInputSize != inputDimensionality && inputDimensionality != 0)
  {
    throw std::logic_error(functionName + ": input size does not match expected"
        " size set with InputDimensions()!");
  }

  // If the input dimensions have not changed from what has been computed
  // before, we can terminate early---the network already has its dimensions
  // set.
  if (inputDimensions == network.InputDimensions())
  {
    inputDimensionsAreSet = true;
    return;
  }

  network.InputDimensions() = inputDimensions;
  network.ComputeOutputDimensions();
  inputDimensionsAreSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType>
std::enable_if_t<
    ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const
{
  if (optimizer.MaxIterations() < samples &&
      optimizer.MaxIterations() != 0)
  {
    Log::Warn << "The optimizer's maximum number of iterations is less than the"
        << " size of the dataset; the optimizer will not pass over the entire "
        << "dataset. To fix this, modify the maximum number of iterations to be"
        << " at least equal to the number of points of your dataset ("
        << samples << ")." << std::endl;
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType>
std::enable_if_t<
    !ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
FFN<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WarnMessageMaxIterations(OptimizerType& /* optimizer */,
                            size_t /* samples */) const
{
  // Nothing to do here.
}

} // namespace mlpack

#endif
