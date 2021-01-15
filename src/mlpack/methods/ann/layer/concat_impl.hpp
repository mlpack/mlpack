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
    const bool model, const bool run) :
    axis(0),
    useAxis(false),
    model(model),
    run(run),
    channels(1)
{
  weights.set_size(0, 0);
}

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::ConcatType(
    arma::Row<size_t>& inputSize,
    const size_t axis,
    const bool model,
    const bool run) :
    inputSize(inputSize),
    axis(axis),
    useAxis(true),
    model(model),
    run(run)
{
  weights.set_size(0, 0);

  // Parameters to help calculate the number of channels.
  size_t oldColSize = 1, newColSize = 1;
  // Axis is specified and useAxis is true.
  if (useAxis)
  {
    // Axis is specified without input dimension.
    // Throw an error.
    if (inputSize.n_elem > 0)
    {
      // Calculate rowSize, newColSize based on the axis
      // of concatenation. Finally concat along cols and
      // reshape to original format i.e. (input, batch_size).
      size_t i = std::min(axis + 1, (size_t) inputSize.n_elem);
      for (; i < inputSize.n_elem; ++i)
        newColSize *= inputSize[i];
    }
    else
      Log::Fatal << "Input dimensions not specified." << std::endl;
  }
  else
    channels = 1;

  if (newColSize <= 0)
    Log::Fatal << "Col size is zero." << std::endl;

  channels = newColSize / oldColSize;
  inputSize.clear();
}

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::~ConcatType()
{
  if (!model)
  {
    // Clear memory.
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->Forward(input,network[i]->OutputParameter());
    }
  }

  output = network.front()->OutputParameter();

  // Reshape output to incorporate the channels.
  output.reshape(output.n_rows / channels, output.n_cols * channels);

  for (size_t i = 1; i < network.size(); ++i)
  {
    OutputType out = network[i]->OutputParameter();

    out.reshape(out.n_rows / channels, out.n_cols * channels);

    // Vertically concatentate output from each layer.
    output = arma::join_cols(output, out);
  }
  // Reshape output to its original shape.
  output.reshape(output.n_rows * channels, output.n_cols / channels);
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  size_t rowCount = 0;
  if (run)
  {
    OutputType delta;
    OutputType gyTmp(((OutputType&) gy).memptr(), gy.n_rows / channels,
        gy.n_cols * channels, false, false);
    for (size_t i = 0; i < network.size(); ++i)
    {
      // Use rows from the error corresponding to the output from each layer.
      size_t rows = network[i]->OutputParameter().n_rows;

      // Extract from gy the parameters for the i-th network.
      delta = gyTmp.rows(rowCount / channels, (rowCount + rows) / channels - 1);
      delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

      network[i]->Backward(
          network[i]->OutputParameter(),
          delta,
          network[i]->Delta()
        );
      rowCount += rows;
    }

    g = network[0]->Delta();
    for (size_t i = 1; i < network.size(); ++i)
    {
      g += network[i]->Delta();
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
    rowCount += network[i]->OutputParameter().n_rows;
  }
  rows = network[index]->OutputParameter().n_rows;

  // Reshape gy to extract the i-th layer gy.
  OutputType gyTmp(((OutputType&) gy).memptr(), gy.n_rows / channels,
      gy.n_cols * channels, false, false);

  OutputType delta = gyTmp.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

  network[index]->Backward(
      network[index]->OutputParameter(),
      delta,
      network[index]->Delta()
    );

  g = network[index]->Delta();
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */)
{
  if (run)
  {
    size_t rowCount = 0;
    // Reshape error to extract the i-th layer error.
    OutputType errorTmp(((OutputType&) error).memptr(),
        error.n_rows / channels, error.n_cols * channels, false, false);
    for (size_t i = 0; i < network.size(); ++i)
    {
      size_t rows = network[i]->OutputParameter().n_rows;

      // Extract from error the parameters for the i-th network.
      OutputType err = errorTmp.rows(rowCount / channels, (rowCount + rows) /
          channels - 1);
      err.reshape(err.n_rows * channels, err.n_cols / channels);

      network[i]->Gradient(input, err, network[i]->Gradient());
      rowCount += rows;
    }
  }
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& /* gradient */,
    const size_t index)
{
  size_t rowCount = 0;
  for (size_t i = 0; i < index; ++i)
  {
    rowCount += network[i]->OutputParameter().n_rows;
  }
  size_t rows = network[index]->OutputParameter().n_rows;

  OutputType errorTmp(((OutputType&) error).memptr(),
      error.n_rows / channels, error.n_cols * channels, false, false);
  OutputType err = errorTmp.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  err.reshape(err.n_rows * channels, err.n_cols / channels);

  network[index]->Gradient(input, err, network[index]->Gradient());
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ConcatType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(run));

  // Do we have to load or save a model?
  if (model)
  {
    // Clear memory first, if needed.
    if (cereal::is_loading<Archive>())
    {
      for (size_t i = 0; i < network.size(); ++i)
        delete network[i];
    }
    // ar(CEREAL_VECTOR_VARIANT_POINTER(network));
  }
}

} // namespace ann
} // namespace mlpack


#endif
