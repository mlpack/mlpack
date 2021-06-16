/**
 * @file methods/ann/layer/concat_impl.hpp
 * @author Marcus Edel
 * @author Mehul Kumar Nirala
 *
 * Implementation of the Concat class, which acts as a concatenation contain.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_IMPL_HPP

// In case it hasn't yet been included.
#include "concat.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::ConcatType(
    const bool run) :
    axis(0),
    useAxis(false),
    channels(1),
    run(run)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::ConcatType(
    const size_t axis,
    const bool run) :
    axis(axis),
    useAxis(true),
    channels(0),
    run(run)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::~ConcatType()
{
  // Clear memory.
  for (size_t i = 0; i < this->network.size(); ++i)
    delete this->network[i];
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  this->InitializeForwardPassMemory();
  ComputeChannels();

  if (run)
  {
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      this->network[i]->Forward(input, this->layerOutputs[i]);
    }
  }

  output = this->layerOutputs.front();

  // Reshape output to incorporate the channels.
  output.reshape(output.n_rows / channels, output.n_cols * channels);

  for (size_t i = 1; i < this->network.size(); ++i)
  {
    OutputType out = this->layerOutputs[i];

    out.reshape(out.n_rows / channels, out.n_cols * channels);

    // Vertically concatenate output from each layer.
    output = arma::join_cols(output, out);
  }

  // Reshape output to its original shape.
  output.reshape(output.n_rows * channels, output.n_cols / channels);
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  this->InitializeBackwardPassMemory();

  size_t rowCount = 0;
  if (run)
  {
    OutputType delta;
    OutputType gyTmp(((OutputType&) gy).memptr(), gy.n_rows / channels,
        gy.n_cols * channels, false, false);
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      // Use rows from the error corresponding to the output from each layer.
      size_t rows = this->network[i]->OutputParameter().n_rows;

      // Extract from gy the parameters for the i-th this->network.
      delta = gyTmp.rows(rowCount / channels, (rowCount + rows) / channels - 1);
      delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

      this->network[i]->Backward(this->layerOutputs[i], delta, this->layerDeltas[i]);
      rowCount += rows;
    }

    g = this->layerDeltas[0];
    for (size_t i = 1; i < this->network.size(); ++i)
    {
      g += this->layerDeltas[i];
    }
  }
  else
    g = gy;
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g,
    const size_t index)
{
  size_t rowCount = 0, rows = 0;

  for (size_t i = 0; i < index; ++i)
  {
    rowCount += this->layerOutputs[i].n_rows;
  }
  rows = this->layerOutputs[index].n_rows;

  // Reshape gy to extract the i-th layer gy.
  OutputType gyTmp(((OutputType&) gy).memptr(), gy.n_rows / channels,
      gy.n_cols * channels, false, false);

  OutputType delta = gyTmp.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

  this->network[index]->Backward(this->layerOutputs[index], delta, this->layerDeltas[index]);

  g = this->layerDeltas[index];
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  if (run)
  {
    size_t rowCount = 0;
    size_t paramCount = 0;
    // Reshape error to extract the i-th layer error.
    OutputType errorTmp(((OutputType&) error).memptr(),
        error.n_rows / channels, error.n_cols * channels, false, false);
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      size_t rows = this->network[i]->OutputParameter().n_rows;

      // Extract from error the parameters for the i-th this->network.
      OutputType err = errorTmp.rows(rowCount / channels, (rowCount + rows) /
          channels - 1);
      err.reshape(err.n_rows * channels, err.n_cols / channels);

      this->network[i]->Gradient(input, err, OutputType(gradient.colptr(paramCount),
          1, this->network[i]->WeightSize(), false, true));
      rowCount += rows;
      paramCount += this->network[i]->WeightSize();
    }
  }
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient,
    const size_t index)
{
  size_t rowCount = 0;
  size_t paramCount = 0;
  for (size_t i = 0; i < index; ++i)
  {
    rowCount += this->network[i]->OutputParameter().n_rows;
    paramCount += this->network[i]->WeightSize();
  }
  size_t rows = this->network[index]->OutputParameter().n_rows;

  OutputType errorTmp(((OutputType&) error).memptr(),
      error.n_rows / channels, error.n_cols * channels, false, false);
  OutputType err = errorTmp.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  err.reshape(err.n_rows * channels, err.n_cols / channels);

  this->network[index]->Gradient(input, err, OutputType(gradient.memptr(), 1,
      paramCount, false, true));
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ConcatType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(axis));
  ar(CEREAL_NVP(useAxis));
  ar(CEREAL_NVP(channels));
  ar(CEREAL_NVP(run));
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::ComputeChannels()
{
  // Parameters to help calculate the number of channels.
  size_t oldColSize = 1, newColSize = 1;
  // Axis is specified and useAxis is true.
  if (useAxis)
  {
    // Axis is specified without input dimension.
    // Throw an error.
    if (inputDimensions.size() > 0)
    {
      // Calculate rowSize, newColSize based on the axis
      // of concatenation. Finally concat along cols and
      // reshape to original format i.e. (input, batch_size).
      size_t i = std::min(axis + 1, inputDimensions.size());
      for (; i < inputDimensions.size(); ++i)
        newColSize *= inputSize[i];
    }
    else
      Log::Fatal << "Concat(): input has zero dimensions." << std::endl;

    if (newColSize <= 0)
      Log::Fatal << "Concat(): column size is zero." << std::endl;

    channels = newColSize / oldColSize;
  }
}

} // namespace ann
} // namespace mlpack


#endif
