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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SpatialDropout<InputDataType, OutputDataType>::SpatialDropout() :
    size(0),
    ratio(0.5),
    scale(1.0 / (1.0 - ratio)),
    reset(false),
    batchSize(0),
    inputSize(0),
    deterministic(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
SpatialDropout<InputDataType, OutputDataType>::SpatialDropout(
    const size_t size,
    const double ratio) :
    size(size),
    ratio(ratio),
    scale(1.0 / (1.0 - ratio)),
    reset(false),
    batchSize(0),
    inputSize(0),
    deterministic(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void SpatialDropout<InputDataType, OutputDataType>::Forward(
  const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (!reset)
  {
    batchSize = input.n_cols;
    inputSize = input.n_rows / size;
    reset = true;
  }

  if (deterministic)
    output = input;
  else
  {
    output.zeros(arma::size(input));
    arma::cube inputTemp(const_cast<arma::mat&>(input).memptr(), inputSize,
        size, batchSize, false, false);
    arma::cube outputTemp(const_cast<arma::mat&>(output).memptr(), inputSize,
        size, batchSize, false, false);
    arma::mat probabilities(1, size);
    arma::mat maskRow(1, size);
    probabilities.fill(ratio);
    ann::BernoulliDistribution<> bernoulli_dist(probabilities, false);
    maskRow = bernoulli_dist.Sample();
    mask = arma::repmat(maskRow, inputSize, 1);

    for (size_t n = 0; n < batchSize; n++)
      outputTemp.slice(n) = inputTemp.slice(n) % mask * scale;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void SpatialDropout<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  g.zeros(arma::size(input));
  arma::cube gyTemp(const_cast<arma::mat&>(gy).memptr(), inputSize, size,
      batchSize, false, false);
  arma::cube gTemp(const_cast<arma::mat&>(g).memptr(), inputSize, size,
      batchSize, false, false);

  for (size_t n = 0; n < batchSize; n++)
    gTemp.slice(n) = gyTemp.slice(n) % mask * scale;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SpatialDropout<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(size);
  ar & BOOST_SERIALIZATION_NVP(ratio);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(inputSize);
  ar & BOOST_SERIALIZATION_NVP(reset);
  ar & BOOST_SERIALIZATION_NVP(deterministic);

  // Reset scale.
  scale = 1.0 / (1.0 - ratio);
}

} // namespace ann
} // namespace mlpack

#endif
