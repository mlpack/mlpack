/**
 * @file methods/ann/layer/spatial_dropout_impl.hpp
 * @author Anjishnu Mukherjee
 *
 * Implementation of the SpatialDropout class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SPATIAL_DROPOUT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SPATIAL_DROPOUT_IMPL_HPP

// In case it hasn't been included yet.
#include "spatial_dropout.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename InputType, typename OutputType>
SpatialDropoutType<InputType, OutputType>::SpatialDropoutType() :
    size(0),
    ratio(0.5),
    scale(1.0 / (1.0 - ratio)),
    reset(false),
    batchSize(0),
    inputSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
SpatialDropoutType<InputType, OutputType>::SpatialDropoutType(
    const size_t size,
    const double ratio) :
    size(size),
    ratio(ratio),
    scale(1.0 / (1.0 - ratio)),
    reset(false),
    batchSize(0),
    inputSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void SpatialDropoutType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  Log::Assert(input.n_rows % size == 0, "Input features must be divisible "
      "by feature maps.");

  if (!reset)
  {
    batchSize = input.n_cols;
    inputSize = input.n_rows / size;
    reset = true;
  }

  if (deterministic)
  {
    output = input;
  }
  else
  {
    output.zeros(arma::size(input));
    arma::Cube<typename InputType::elem_type> inputTemp(
        const_cast<InputType&>(input).memptr(), inputSize, size, batchSize,
        false, true);
    arma::Cube<typename OutputType::elem_type> outputTemp(
        const_cast<OutputType&>(output).memptr(), inputSize, size, batchSize,
        false, true);
    OutputType probabilities(1, size);
    OutputType maskRow(1, size);
    probabilities.fill(ratio);
    BernoulliDistribution<> bernoulli_dist(probabilities, false);
    maskRow = bernoulli_dist.Sample();
    mask = repmat(maskRow, inputSize, 1);

    for (size_t n = 0; n < batchSize; n++)
      outputTemp.slice(n) = inputTemp.slice(n) % mask * scale;
  }
}

template<typename InputType, typename OutputType>
void SpatialDropoutType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  g.zeros(arma::size(input));
  arma::Cube<typename OutputType::elem_type> gyTemp(
      const_cast<OutputType&>(gy).memptr(), inputSize, size, batchSize, false,
      true);
  arma::Cube<typename OutputType::elem_type> gTemp(
      const_cast<OutputType&>(g).memptr(), inputSize, size, batchSize, false,
      true);

  for (size_t n = 0; n < batchSize; n++)
    gTemp.slice(n) = gyTemp.slice(n) % mask * scale;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void SpatialDropoutType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(ratio));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(inputSize));
  ar(CEREAL_NVP(reset));

  // Reset scale.
  if (Archive::is_loading::value)
    scale = 1.0 / (1.0 - ratio);
}

} // namespace mlpack

#endif
