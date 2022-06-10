/**
 * @file methods/ann/layer/lambda_map_reduce_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of the base class for neural network layers that are wrappers
 * around other layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAMBDA_MAP_REDUCE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LAMBDA_MAP_REDUCE_IMPL_HPP

#include "lambda_map_reduce.hpp"

namespace mlpack {
namespace ann {

template<typename ReductionRuleType, typename MatType>
LambdaMapReduceType<ReductionRuleType, MatType>::LambdaMapReduceType() :
    Layer<MatType>(),
    inSize(0),
    totalInputSize(0),
    totalOutputSize(0)
{
  // Nothing to do.
}

template<typename ReductionRuleType, typename MatType>
LambdaMapReduceType<ReductionRuleType, MatType>::LambdaMapReduceType(const LambdaMapReduceType& other) :
    Layer<MatType>(other),
    inSize(other.inSize),
    totalInputSize(other.totalInputSize),
    totalOutputSize(other.totalOutputSize),
    reductionRule(other.reductionRule),
    layerOutputMatrix(other.layerOutputMatrix),
    layerDeltaMatrix(other.layerDeltaMatrix)
{
  // Copy each layer.
  for (size_t i = 0; i < other.network.size(); ++i)
    network.push_back(other.network[i]->Clone());

  // Ensure that the aliases for layers during passes have the right size.
  layerOutputs.resize(network.size(), MatType());
  layerDeltas.resize(network.size(), MatType());
  layerGradients.resize(network.size(), MatType());

  // layerOutputs, layerDeltas, and layerGradients will be reset the next time
  // Forward(), Backward(), or Gradient() is called.
}

template<typename ReductionRuleType, typename MatType>
LambdaMapReduceType<ReductionRuleType, MatType>::LambdaMapReduceType(LambdaMapReduceType&& other) :
    Layer<MatType>(other),
    network(std::move(other.network)),
    inSize(std::move(other.inSize)),
    totalInputSize(std::move(other.totalInputSize)),
    totalOutputSize(std::move(other.totalOutputSize)),
    reductionRule(std::move(other.reductionRule)),
    layerOutputMatrix(std::move(other.layerOutputMatrix)),
    layerDeltaMatrix(std::move(other.layerDeltaMatrix))
{
  // Ensure that the aliases for layers during passes have the right size.
  layerOutputs.resize(network.size(), MatType());
  layerDeltas.resize(network.size(), MatType());
  layerGradients.resize(network.size(), MatType());

  // layerOutputs, layerDeltas, and layerGradients will be reset the next time
  // Forward(), Backward(), or Gradient() is called.

  other.layerOutputs.clear();
  other.layerDeltas.clear();
  other.layerGradients.clear();
}

template<typename ReductionRuleType, typename MatType>
LambdaMapReduceType<ReductionRuleType, MatType>& LambdaMapReduceType<ReductionRuleType, MatType>::operator=(const LambdaMapReduceType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);

    network.clear();
    layerOutputs.clear();
    layerDeltas.clear();
    layerGradients.clear();

    inSize = other.inSize;
    totalInputSize = other.totalInputSize;
    totalOutputSize = other.totalOutputSize;
    reductionRule = other.reductionRule;

    layerOutputMatrix = other.layerOutputMatrix;
    layerDeltaMatrix = other.layerDeltaMatrix;

    for (size_t i = 0; i < other.network.size(); ++i)
      network.push_back(other.network[i]->Clone());

    // Ensure that the aliases for layers during passes have the right size.
    layerOutputs.resize(network.size(), MatType());
    layerDeltas.resize(network.size(), MatType());
    layerGradients.resize(network.size(), MatType());
  }

  return *this;
}

template<typename ReductionRuleType, typename MatType>
LambdaMapReduceType<ReductionRuleType, MatType>& LambdaMapReduceType<ReductionRuleType, MatType>::operator=(LambdaMapReduceType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);

    layerOutputs.clear();
    layerDeltas.clear();
    layerGradients.clear();

    inSize = std::move(other.inSize);
    totalInputSize = std::move(other.totalInputSize);
    totalOutputSize = std::move(other.totalOutputSize);
    reductionRule = std::move(other.reductionRule);

    network = std::move(other.network);

    layerOutputs.resize(network.size(), MatType());
    layerDeltas.resize(network.size(), MatType());
    layerGradients.resize(network.size(), MatType());

    other.layerOutputs.clear();
    other.layerDeltas.clear();
    other.layerGradients.clear();
  }

  return *this;
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::Forward(
    const MatType& input, MatType& output)
{
  Forward(input, output, 0, network.size() - 1);
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::Forward(
    const MatType& input,
    MatType& output,
    const size_t start,
    const size_t end)
{
  // Make sure training/testing mode is set right in each layer.
  for (size_t i = 0; i < network.size(); ++i)
    network[i]->Training() = this->training;

  // Note that we use `output` for the last layer; layerOutputs is only used for
  // intermediate values between layers.
  if ((end - start) > 0)
  {
    // Initialize memory for the forward pass (if needed).
    InitializeForwardPassMemory(input.n_cols);

    for (size_t i = start; i <= end; i++)
      network[i]->Forward(input, layerOutputs[i]);

		// Reduce the outputs to single output.
    reductionRule.Reduce(layerOutputs, output);
  }
  else if ((end - start) == 0 && network.size() > 0)
  {
    network[start]->Forward(input, output);
  }
  else
  {
    // Empty network?
    output = input;
  }
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  if (network.size() > 1)
  {
    // Initialize memory for the backward pass (if needed).
    InitializeBackwardPassMemory(input.n_cols);

		std::vector<MatType> layerTempDeltas;
		// Compute the gy for all layers.
    reductionRule.UnReduce(gy, network.size(), layerTempDeltas);

    g.zeros();
    for (size_t i = network.size() - 1; i >= 0; i--) {
      network[i]->Backward(layerOutputs[i], layerTempDeltas[i], layerDeltas[i]);
			g += layerDeltas[i];
		}
  }
  else if (network.size() == 1)
  {
    network[0]->Backward(input, gy, g);
  }
  else
  {
    // Empty network?
    g = gy;
  }
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::Gradient(
    const MatType& input, const MatType& error, MatType& gradient)
{
  // We assume gradient has the right size already.

  // Pass gradients through each layer.
  if (network.size() > 1)
  {
    // Initialize memory for the gradient pass (if needed).
    InitializeGradientPassMemory(gradient);

    std::vector<MatType> layerTempDeltas;
    // Compute the error for all layers.
    reductionRule.UnReduce(error, network.size(), layerTempDeltas);

    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Gradient(input, layerTempDeltas[i],
          layerGradients[i]);
    }
  }
  else if (network.size() == 1)
  {
    network[0]->Gradient(input, error, gradient);
  }
  else
  {
    // Nothing to do if the network is empty... there is no gradient.
  }
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::SetWeights(typename MatType::elem_type* weightsPtr)
{
  size_t start = 0;
  const size_t totalWeightSize = WeightSize();
  for (size_t i = 0; i < network.size(); ++i)
  {
    const size_t weightSize = network[i]->WeightSize();

    // Sanity check: ensure we aren't passing memory past the end of the
    // parameters.
    Log::Assert(start + weightSize <= totalWeightSize,
        "FNN::SetLayerMemory(): parameter size does not match total layer "
        "weight size!");

    network[i]->SetWeights(weightsPtr + start);
    start += weightSize;
  }

  // Technically this check should be unnecessary, but there's nothing wrong
  // with a little paranoia...
  Log::Assert(start == totalWeightSize,
      "FNN::SetLayerMemory(): total layer weight size does not match parameter "
      "size!");

  MakeAlias(weights, weightsPtr, totalWeightSize, 1);
}

template<typename ReductionRuleType, typename MatType>
size_t LambdaMapReduceType<ReductionRuleType, MatType>::WeightSize() const
{
  // Sum the weights in each layer.
  size_t total = 0;
  for (size_t i = 0; i < network.size(); ++i)
    total += network[i]->WeightSize();
  return total;
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::ComputeOutputDimensions()
{
  inSize = 0;
  totalInputSize = 0;
  totalOutputSize = 0;

  // Propagate the input dimensions forward to the output.
  if (network.size() == 0)
  {
    this->outputDimensions = this->inputDimensions;
    return;
  }
  inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
    inSize *= this->inputDimensions[i];
  totalInputSize = network.size() * inSize;

  for (size_t i = 0; i < network.size(); ++i)
  {
    network[i]->InputDimensions() = this->inputDimensions;
    size_t layerOutputSize = network[i]->OutputSize();
    Log::Info << layerOutputSize << std::endl;
    totalOutputSize += layerOutputSize;
  }

	// Compute the output size of the network using reduction rules.
  this->outputDimensions = reductionRule.ReduceSize(network);
}

template<typename ReductionRuleType, typename MatType>
double LambdaMapReduceType<ReductionRuleType, MatType>::Loss() const
{
  double loss = 0.0;
  for (size_t i = 0; i < network.size(); ++i)
    loss += network[i]->Loss();

  return loss;
}

template<typename ReductionRuleType, typename MatType>
template<typename Archive>
void LambdaMapReduceType<ReductionRuleType, MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_VECTOR_POINTER(network));
  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(totalInputSize));
  ar(CEREAL_NVP(totalOutputSize));

  if (Archive::is_loading::value)
  {
    layerOutputMatrix.clear();
    layerDeltaMatrix.clear();
    layerGradients.clear();
    layerOutputs.resize(network.size(), MatType());
    layerDeltas.resize(network.size(), MatType());
    layerGradients.resize(network.size(), MatType());
  }
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::InitializeForwardPassMemory(const size_t batchSize)
{
  // We need to initialize memory to store the output of each layer's Forward()
  // call.  We'll do this all in one matrix, but, the size of this matrix
  // depends on the batch size we are using for computation.  We avoid resizing
  // layerOutputMatrix down, unless we only need 10% or less of it.
  if (batchSize * totalOutputSize > layerOutputMatrix.n_elem ||
      batchSize * totalOutputSize <
          std::floor(0.1 * layerOutputMatrix.n_elem))
  {
    // All outputs will be represented by one big block of memory.
    layerOutputMatrix = MatType(1, batchSize * totalOutputSize);
  }

  // Now, create an alias to the right place for each layer.  We assume that
  // layerOutputs is already sized correctly (this should be done by Add()).
  size_t start = 0;
  for (size_t i = 0; i < layerOutputs.size(); ++i)
  {
    const size_t layerOutputSize = network[i]->OutputSize();
    MakeAlias(layerOutputs[i], layerOutputMatrix.colptr(start),
        layerOutputSize, batchSize);
    start += batchSize * layerOutputSize;
  }
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::InitializeBackwardPassMemory(
    const size_t batchSize)
{
  // We need to initialize memory to store the output of each layer's Backward()
  // call.  We do this similarly to InitializeForwardPassMemory(), but we must
  // store a matrix to use as the delta for each layer.
  if (batchSize * totalInputSize > layerDeltaMatrix.n_elem ||
      batchSize * totalInputSize < std::floor(0.1 * layerDeltaMatrix.n_elem))
  {
    // All deltas will be represented by one big block of memory.
    layerDeltaMatrix = MatType(1, batchSize * totalInputSize);
  }

  // Now, create an alias to the right place for each layer.  We assume that
  // layerDeltas is already sized correctly (this should be done by Add()).
  size_t start = 0;
  for (size_t i = 0; i < layerDeltas.size(); ++i)
  {
    size_t layerInputSize = inSize;
    MakeAlias(layerDeltas[i], layerDeltaMatrix.colptr(start), layerInputSize,
        batchSize);
    start += batchSize * layerInputSize;
  }
}

template<typename ReductionRuleType, typename MatType>
void LambdaMapReduceType<ReductionRuleType, MatType>::InitializeGradientPassMemory(MatType& gradient)
{
  // We need to initialize memory to store the gradients of each layer.  To do
  // this, we need to know the weight size of each layer.
  size_t gradientStart = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    const size_t weightSize = network[i]->WeightSize();
    MakeAlias(layerGradients[i], gradient.memptr() + gradientStart,
        weightSize, 1);
    gradientStart += weightSize;
  }
}

} // namespace ann
} // namespace mlpack

#endif
