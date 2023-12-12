/**
 * @file methods/ann/layer/repeat_impl.hpp
 * @author Adam Kropp
 *
 * Implementation of the Repeat class, which repeats the input n times
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPEAT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_REPEAT_IMPL_HPP

// In case it hasn't yet been included.
#include "repeat.hpp"

namespace mlpack {

template<typename MatType>
RepeatType<MatType>::RepeatType(
    std::vector<size_t> _multiples) :
    Layer<MatType>(),
    multiples(_multiples)
{
  // Nothing to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType() :
    Layer<MatType>()
{
  // Nothing to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType(const RepeatType& other) :
    Layer<MatType>(other),
    multiples(other.multiples),
    outIdxs(other.outIdxs),
    coefs(other.coefs)
{
  // Nothing else to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType(RepeatType&& other) :
    Layer<MatType>(std::move(other)),
    multiples(other.multiples),
    outIdxs(other.outIdxs),
    coefs(other.coefs)
{
  // Nothing else to do.
}

template<typename MatType>
RepeatType<MatType>& RepeatType<MatType>::operator=(const RepeatType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    multiples = other.multiples;
    outIdxs = other.outIdxs;
    coefs = other.coefs;
  }

  return *this;
}

template<typename MatType>
RepeatType<MatType>& RepeatType<MatType>::operator=(RepeatType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    multiples = std::move(other.multiples);
    outIdxs = std::move(other.outIdxs);
    coefs = std::move(other.coefs);
  }

  return *this;
}

template<typename MatType>
void RepeatType<MatType>::Forward(const MatType& input, MatType& output)
{
  output = input.rows(outIdxs);
}

template<typename MatType>
void RepeatType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = coefs * gy;
}

template<typename MatType>
template<typename Archive>
void RepeatType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(multiples));
  ar(CEREAL_NVP(outIdxs));
  ar(CEREAL_NVP(coefs));
}

} // namespace mlpack

#endif
