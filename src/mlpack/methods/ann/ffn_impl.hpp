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
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::FFN(OutputLayerType outputLayer, InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    training(false),
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false),
    totalInputSize(0),
    totalOutputSize(0)
{
  /* Nothing to do here. */
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::FFN(const FFN& network):
    outputLayer(network.outputLayer),
    initializeRule(network.initializeRule),
    parameters(network.parameters),
    inputDimensions(network.inputDimensions),
    predictors(network.predictors),
    responses(network.responses),
    error(network.error),
    training(network.training),
    // These will be set correctly in the first Forward() call.
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false),
    totalInputSize(0),
    totalOutputSize(0)
{
  // Build new layers according to source network
  for (size_t i = 0; i < network.network.size(); ++i)
  {
    this->network.push_back(network.network[i]->Clone());
  }

  // Initialize cached matrix vectors to the right size.
  layerOutputs.resize(this->network.size(), OutputType());
  layerDeltas.resize(this->network.size(), OutputType());
  layerGradients.resize(this->network.size(), OutputType());
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::FFN(FFN&& network):
    outputLayer(std::move(network.outputLayer)),
    initializeRule(std::move(network.initializeRule)),
    network(std::move(network.network)),
    parameters(std::move(network.parameters)),
    inputDimensions(std::move(network.inputDimensions)),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    error(std::move(network.error)),
    training(std::move(network.training)),
    // Aliases will not be correct after a std::move(), so we will manually
    // reset them.
    layerMemoryIsSet(false),
    inputDimensionsAreSet(std::move(network.inputDimensionsAreSet)),
    totalInputSize(std::move(network.totalInputSize)),
    totalOutputSize(std::move(network.totalOutputSize))
{
  // Initialize cached matrix parameters to the right size.
  layerOutputs.resize(this->network.size(), OutputType());
  layerDeltas.resize(this->network.size(), OutputType());
  layerGradients.resize(this->network.size(), OutputType());
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>& FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::operator=(FFN network)
{
  Swap(network);
  return *this;
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::~FFN()
{
  for (size_t i = 0; i < network.size(); ++i)
    delete network[i];
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::ResetData(InputType predictors, InputType responses)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  // Set the network to training mode.
  SetNetworkMode(true);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>::value,
      void>::type
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const
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
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::WarnMessageMaxIterations(OptimizerType& /* optimizer */,
                            size_t /* samples */) const
{
  // Nothing to do here.
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType, typename... CallbackTypes>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Train(InputType predictors,
         InputType responses,
         OptimizerType& optimizer,
         CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  CheckNetwork("FFN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameters, callbacks...);
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
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Train(InputType predictors,
         InputType responses,
         CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  OptimizerType optimizer;

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  CheckNetwork("FFN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameters, callbacks...);
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
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType
>::Forward(const PredictorsType& inputs, ResponsesType& results)
{
  Forward(inputs, results, 0, network.size() - 1);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Forward(const PredictorsType& inputs,
           ResponsesType& results,
           const size_t begin,
           const size_t end)
{
  // Sanity checking...
  if (end < begin)
    return;

  // Ensure the network is valid.
  CheckNetwork("FFN::Forward()", inputs.n_rows);

  // Next we need to ensure that space for layerOutputs is allocated correctly.
  InitializeForwardPassMemory(inputs.n_cols);

  // Ensure that the results matrix is the right size.
  results.set_size(inputs.n_cols * network[end]->OutputSize());

  if (end > begin)
  {
    network[begin]->Forward(inputs, layerOutputs[begin]);

    for (size_t i = 1; i < end - begin; ++i)
    {
      network[begin + i]->Forward(layerOutputs[begin + i - 1],
          layerOutputs[begin + i]);
    }

    network[end]->Forward(layerOutputs[end - 1], results);
  }
  else
  {
    network[end]->Forward(inputs, results);
  }

  // If we computed the pass through all layers, and `results` is not the same
  // matrix as `layerOutputs.back()`, we must ensure that we cache a copy of it,
  // since Backward() may be called later.
  // TODO: clarify this in documentation...
  if (end == (layerOutputs.size() - 1) &&
      results.memptr() != layerOutputs.back().memptr())
  {
    layerOutputs.back() = results;
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename TargetsType, typename GradientsType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Backward(const PredictorsType& inputs,
            const TargetsType& targets,
            GradientsType& gradients)
{
  // We assume that Forward() has already been called!
  double res = outputLayer.Forward(layerOutputs[network.size() - 1], targets);

  for (size_t i = 0; i < network.size(); ++i)
    res += network[i]->Loss();

  // error's size will be set correctly by outputLayer.Backward().
  outputLayer.Backward(layerOutputs[network.size() - 1], targets, error);

  // This does nothing if the gradient's size is already set right.
  gradients.zeros(parameters.n_rows, parameters.n_cols);

  Backward();
  Gradient(inputs, gradients);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Predict(InputType predictors, OutputType& results, const size_t batchSize)
{
  // Ensure that the network is configured correctly.
  CheckNetwork("FFN::Predict()", predictors.n_rows, true, false);

  results.set_size(network.back()->OutputSize(), predictors.n_cols);

  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    InputType predictorAlias(predictors.colptr(i), predictors.n_rows,
        effectiveBatchSize, false, true);
    OutputType resultAlias(results.colptr(i), results.n_rows,
        effectiveBatchSize, false, true);

    Forward(predictorAlias, resultAlias);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(const PredictorsType& predictors, const ResponsesType& responses)
{
  // Sanity check: ensure network is valid.
  CheckNetwork("FFN::Evaluate()", predictors.n_rows);

  // Note that layerOutputs may not yet be initialized correctly, but if it
  // isn't, this will be handled by Forward().
  Forward(predictors, layerOutputs.back());

  double res = outputLayer.Forward(layerOutputs.back(), responses);
  for (size_t i = 0; i < network.size(); ++i)
    res += LossUpdate(network[i]);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(const OutputType& parameters)
{
  CheckNetwork("FFN::Evaluate()", predictors.n_rows, true, false);

  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += Evaluate(parameters, i, 1);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(const OutputType& /* parameters */,
            const size_t begin,
            const size_t batchSize)
{
  Forward(predictors.cols(begin, begin + batchSize - 1), layerOutputs.back());

  double res = outputLayer.Forward(layerOutputs.back(),
      responses.cols(begin, begin + batchSize - 1));

  for (size_t i = 0; i < network.size(); ++i)
    res += network[i]->Loss();

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::EvaluateWithGradient(const OutputType& parameters, OutputType& gradient)
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
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::EvaluateWithGradient(const OutputType& parameters,
                        const size_t begin,
                        OutputType& gradient,
                        const size_t batchSize)
{
  double res = Evaluate(parameters, begin, batchSize);

  for (size_t i = 0; i < network.size(); ++i)
    res += network[i]->Loss();

  outputLayer.Backward(layerOutputs.back(),
      responses.cols(begin, begin + batchSize - 1),
      error);

  Backward();
  Gradient(predictors.cols(begin, begin + batchSize - 1), gradient);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Gradient(const OutputType& parameters,
            const size_t begin,
            OutputType& gradient,
            const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Reset(const size_t inputDimensionality)
{
  parameters.clear();

  if (inputDimensionality != 0)
  {
    CheckNetwork("FFN::Reset()", inputDimensionality, true, false);
  }
  else
  {
    const size_t inputDims = std::accumulate(inputDimensions.begin(),
        inputDimensions.end(), 0);
    CheckNetwork("FFN::Reset()", inputDims, true, false);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::InitializeWeights()
{
  // Set the network to testing mode.
  SetNetworkMode(false);

  // Reset the network parameters with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network, parameters);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::SetLayerMemory()
{
  size_t totalWeightSize = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    const size_t weightSize = network[i]->WeightSize();

    // Sanity check: ensure we aren't passing memory past the end of the
    // parameters.
    Log::Assert(totalWeightSize + weightSize <= parameters.n_elem,
        "FNN::SetLayerMemory(): parameter size does not match total layer "
        "weight size!");

    network[i]->SetWeights(parameters.memptr() + totalWeightSize);
    totalWeightSize += weightSize;
  }

  Log::Assert(totalWeightSize == parameters.n_elem,
      "FNN::SetLayerMemory(): total layer weight size does not match parameter "
      "size!");

  layerMemoryIsSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::CheckNetwork(const std::string& functionName,
                const size_t inputDimensionality,
                const bool setMode,
                const bool training)
{
  // If the network is empty, we can't do anything.
  if (network.size() == 0)
  {
    throw std::invalid_argument(functionName + ": cannot use network with no "
        "layers!");
  }

  // Next, check that the input dimensions for each layer are correct.  Note
  // that this will throw an exception if the user has passed data that does not
  // match this->inputDimensions.
  if (!inputDimensionsAreSet)
    UpdateDimensions(functionName, inputDimensionality);

  // We may need to initialize the `parameters` matrix if it is empty.
  if (parameters.is_empty())
    InitializeWeights();

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
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::UpdateDimensions(const std::string& functionName,
                    const size_t inputDimensionality)
{
  // If the input dimensions are completely unset, then assume our input is
  // flat.
  if (inputDimensions.size() == 0)
    inputDimensions = { inputDimensionality };

  totalInputSize = std::accumulate(inputDimensions.begin(),
      inputDimensions.end(), 0);

  // TODO: improve this error message...
  if (totalInputSize != inputDimensionality)
  {
    throw std::logic_error(functionName + ": input size does not match expected"
        " size set with InputDimensions()!");
  }

  // If the input dimensions have not changed from what has been computed
  // before, we can terminate early---the network already has its dimensions
  // set.
  if (inputDimensions == network.front()->InputDimensions())
  {
    inputDimensionsAreSet = true;
    return;
  }

  network.front()->InputDimensions() = inputDimensions;
  totalInputSize += network[0]->OutputSize();
  totalOutputSize = network[0]->OutputSize();
  for (size_t i = 1; i < network.size(); ++i)
  {
    network[i]->InputDimensions() = network[i - 1]->OutputDimensions();
    const size_t layerOutputSize = network[i]->OutputSize();

    // If we are not at the last layer, then this output is the input to the
    // next layer.
    if (i != network.size() - 1)
      totalInputSize += layerOutputSize;

    totalOutputSize += layerOutputSize;
  }

  inputDimensionsAreSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::SetNetworkMode(const bool training)
{
  this->training = training;
  for (size_t i = 0; i < network.size(); ++i)
    network[i]->Training() = this->training;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Backward()
{
  // This actually does the backward computation.  In order to do that, we need
  // to make sure that the local variables we'll be using are properly
  // initialized.
  InitializeBackwardPassMemory(layerOutputs.back().n_cols);

  // layerDeltas.back() will have size
  // network[network.size() - 2]->OutputSize() * batchSize.
  network.back()->Backward(layerOutputs.back(), error, layerDeltas.back());

  for (size_t i = 2; i <= network.size(); ++i)
  {
    network[network.size() - i]->Backward(
        layerOutputs[network.size() - i],
        layerDeltas[network.size() - i + 1],
        layerDeltas[network.size() - i]);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Gradient(const InputType& input, OutputType& gradient)
{
  // Make sure that the memory is initialized for layerGradients.
  InitializeGradientPassMemory(gradient);

  network.front()->Gradient(input, layerDeltas[1], layerGradients.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    network[i]->Gradient(layerOutputs[i - 1], layerDeltas[i + 1],
        layerGradients[i]);
  }

  network.back()->Gradient(layerOutputs[network.size() - 2], error,
      layerGradients.back());
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename Archive>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  // Serialize the output layer and initialization rule.
  ar(CEREAL_NVP(outputLayer));
  ar(CEREAL_NVP(initializeRule));

  // Serialize the network itself.
  ar(CEREAL_VECTOR_POINTER(network));
  ar(CEREAL_NVP(parameters));

  // Serialize the expected input size.
  ar(CEREAL_NVP(inputDimensions));
  ar(CEREAL_NVP(training));

  ar(CEREAL_NVP(totalInputSize));
  ar(CEREAL_NVP(totalOutputSize));

  // If we are loading, we need to initialize the weights.
  if (cereal::is_loading<Archive>())
  {
    // We can clear these members, since it's not possible to serialize in the
    // middle of training and resume.
    predictors.clear();
    responses.clear();

    layerOutputMatrix.clear();
    layerOutputs.clear();
    layerOutputs.resize(network.size(), OutputType());

    deltaMatrix.clear();
    layerDeltas.clear();
    layerDeltas.resize(network.size(), OutputType());

    layerGradients.clear();
    layerGradients.resize(network.size(), OutputType());

    layerMemoryIsSet = false;
    inputDimensionsAreSet = false;

    // The weights in parameters will be correctly set for each layer in the
    // first call to Forward().
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Swap(FFN& network)
{
  std::swap(outputLayer, network.outputLayer);
  std::swap(initializeRule, network.initializeRule);
  std::swap(this->network, network.network);
  std::swap(parameters, network.parameters);
  std::swap(inputDimensions, network.inputDimensions);
  std::swap(predictors, network.predictors);
  std::swap(responses, network.responses);
  std::swap(error, network.error);
  std::swap(training, network.training);
  std::swap(inputDimensionsAreSet, network.inputDimensionsAreSet);
  std::swap(totalInputSize, network.totalInputSize);
  std::swap(totalOutputSize, network.totalOutputSize);
  std::swap(layerOutputMatrix, network.layerOutputMatrix);
  std::swap(layerOutputs, network.layerOutputs);
  std::swap(deltaMatrix, network.deltaMatrix);
  std::swap(layerDeltas, network.layerDeltas);
  std::swap(layerGradients, network.layerGradients);

  // std::swap() will not preserve Armadillo aliases correctly, so we will reset
  // those.
  layerMemoryIsSet = false;
  layerOutputs.resize(this->network.size(), OutputType());
  layerDeltas.resize(this->network.size(), OutputType());
  layerGradients.resize(this->network.size(), OutputType());

  network.layerMemoryIsSet = false;
  network.layerOutputs.resize(network.network.size(), OutputType());
  network.layerDeltas.resize(network.network.size(), OutputType());
  network.layerGradients.resize(network.network.size(), OutputType());
};

// Initialize memory to be used for storing the outputs of each layer, if
// necessary.  This should be called at the start of each forward pass, but does
// not need to be called at any other time.
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::InitializeForwardPassMemory(const size_t batchSize)
{
  // We need to initialize memory to store the output of each layer's Forward()
  // call.  We'll do this all in one matrix, but, the size of this matrix
  // depends on the batch size we are using for computation.  We avoid resizing
  // layerOutputMatrix down, unless we only need 10% or less of it.
  if (batchSize * totalOutputSize > layerOutputMatrix.n_elem ||
      batchSize * totalOutputSize < std::floor(0.1 * layerOutputMatrix.n_elem))
  {
    // All outputs will be represented by one big block of memory.
    layerOutputMatrix = OutputType(1, batchSize * totalOutputSize);
  }

  // Now, create an alias to the right place for each layer.  We assume that
  // layerOutputs is already sized correctly (this should be done by Add()).
  size_t start = 0;
  for (size_t i = 0; i < layerOutputs.size(); ++i)
  {
    const size_t layerOutputSize = network[i]->OutputSize();

    Log::Assert(
        start + layerOutputSize * batchSize <= batchSize * totalOutputSize,
        "Creating alias outside bounds!");

    MakeAlias(layerOutputs[i], layerOutputMatrix.memptr() + start,
        layerOutputSize, batchSize);
    start += batchSize * layerOutputSize;
  }
}

// Initialize memory to be used for the backward pass.  This should be called at
// the start of each backward pass, but does not need to be called at any other
// time.
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::InitializeBackwardPassMemory(const size_t batchSize)
{
  // We need to initialize memory to store the output of each layer's Backward()
  // and Gradient() calls.  We do this similarly to
  // InitializeForwardPassMemory(), but we must store a matrix to use as the
  // delta for each layer.
  const size_t inputSize = std::accumulate(inputDimensions.begin(),
      inputDimensions.end(), 0);
  if (batchSize * totalInputSize > deltaMatrix.n_elem ||
      batchSize * totalInputSize < std::floor(0.1 * layerOutputMatrix.n_elem))
  {
    // All deltas will be represented by one big block of memory.
    deltaMatrix = OutputType(1, batchSize * totalInputSize);
  }

  // Now, create an alias to the right place for each layer.  We assume that
  // layerDeltas is already sized correctly (this should be done by Add()).
  size_t start = 0;
  for (size_t i = 0; i < layerDeltas.size(); ++i)
  {
    const size_t layerInputSize = (i == 0) ? inputSize :
        network[i - 1]->OutputSize();

    Log::Assert(
        start + layerInputSize * batchSize <= batchSize * totalInputSize,
        "Creating alias outside bounds!");

    MakeAlias(layerDeltas[i], deltaMatrix.memptr() + start, layerInputSize,
        batchSize);
    start += batchSize * layerInputSize;
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::InitializeGradientPassMemory(OutputType& gradient)
{
  // Sanity check: make sure the gradient has the right size.
  size_t totalWeightSize = 0;
  for (size_t i = 0; i < layerGradients.size(); ++i)
    totalWeightSize += network[i]->WeightSize();
  // TODO: cache totalWeightSize ...
  if (gradient.n_elem != totalWeightSize)
    gradient.set_size(totalWeightSize);

  // We need to initialize the aliases `layerGradients()` for each layer.
  size_t start = 0;
  for (size_t i = 0; i < layerGradients.size(); ++i)
  {
    const size_t layerParamSize = network[i]->WeightSize();
    MakeAlias(layerGradients[i], gradient.memptr() + start, layerParamSize, 1);
    start += layerParamSize;
  }
}

// Utility function to make an alias.
template<typename MatType>
void MakeAlias(MatType& m,
               typename MatType::elem_type* newMem,
               const size_t numRows,
               const size_t numCols)
{
  // We use placement new to reinitialize the object, since the copy and move
  // assignment operators in Armadillo will end up copying memory instead of
  // making an alias.
  m.~MatType();
  new (&m) MatType(newMem, numRows, numCols, false, true);
}

} // namespace ann
} // namespace mlpack

#endif
