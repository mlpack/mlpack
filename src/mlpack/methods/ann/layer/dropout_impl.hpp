/**
 * @file dropout_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Dropout class, which implements a regularizer that
 * randomly sets units to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPOUT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPOUT_IMPL_HPP

// In case it hasn't yet been included.
#include "dropout.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Dropout<InputDataType, OutputDataType>::Dropout(
    const double ratio, const bool rescale) :
    ratio(ratio),
    scale(1.0 / (1.0 - ratio)),
    rescale(rescale)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Dropout<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& output)
{
  // The dropout mask will not be multiplied in the deterministic mode
  // (during testing).
  if (deterministic)
  {
    if (!rescale)
    {
      output = input;
    }
    else
    {
      output = input * scale;
    }
  }
  else
  {
    // Scale with input / (1 - ratio) and set values to zero with probability
    // ratio.
    mask = arma::randu<arma::Mat<eT> >(input.n_rows, input.n_cols);
    mask.transform( [&](double val) { return (val > ratio); } );
    output = input % mask * scale;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Dropout<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  g = gy % mask * scale;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Dropout<InputDataType, OutputDataType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(ratio, "ratio");
  ar & data::CreateNVP(rescale, "rescale");
}

} // namespace ann
} // namespace mlpack

#endif
