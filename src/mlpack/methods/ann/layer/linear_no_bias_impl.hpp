/**
 * @file methods/ann/layer/linear_no_bias_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LinearNoBias class also known as fully-connected layer
 * or affine transformation without the bias term.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP

// In case it hasn't yet been included.
#include "linear_no_bias.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
LinearNoBias<InputDataType, OutputDataType, RegularizerType>::LinearNoBias() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
LinearNoBias<InputDataType, OutputDataType, RegularizerType>::LinearNoBias(
    const size_t inSize,
    const size_t outSize,
    RegularizerType regularizer) :
    inSize(inSize),
    outSize(outSize),
    regularizer(regularizer)
{
  weights.set_size(outSize * inSize, 1);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Reset()
{
  weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  output = weight * input;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  g = weight.t() * gy;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      error * input.t());
  regularizer.Evaluate(weights, gradient);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename Archive>
void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(inSize);
  ar & CEREAL_NVP(outSize);

  // This is inefficient, but necessary so that WeightSetVisitor sets the right
  // size.
  if (Archive::is_loading::value)
    weights.set_size(outSize * inSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
