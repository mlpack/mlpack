/**
 * @file methods/ann/layer/join_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Join module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP

// In case it hasn't yet been included.
#include "join.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
JoinType<InputType, OutputType>::JoinType() :
    inSizeRows(0),
    inSizeCols(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void JoinType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  inSizeRows = input.n_rows;
  inSizeCols = input.n_cols;
  output = vectorise(input);
}

template<typename InputType, typename OutputType>
void JoinType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  g = OutputType(((OutputType&) gy).memptr(), inSizeRows, inSizeCols, false,
      false);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void JoinType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(inSizeRows));
  ar(CEREAL_NVP(inSizeCols));
}

} // namespace mlpack

#endif
