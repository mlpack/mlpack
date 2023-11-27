/**
 * @file methods/ann/layer/replicate_impl.hpp
 * @author Adam Kropp
 *
 * Implementation of the Replicate class, which replicates the input n times
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPLICATE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_REPLICATE_IMPL_HPP

// In case it hasn't yet been included.
#include "replicate.hpp"

namespace mlpack {

template<typename MatType>
ReplicateType<MatType>::ReplicateType(
    std::vector<size_t> _multiples) :
    Layer<MatType>(),
    multiples(_multiples)
{
  // Nothing to do.
}

template<typename MatType>
ReplicateType<MatType>::ReplicateType() :
    Layer<MatType>()
{
  // Nothing to do.
}

template<typename MatType>
ReplicateType<MatType>::~ReplicateType()
{
  // Nothing to do: the child layer memory is already cleared by MultiLayer.
}

template<typename MatType>
ReplicateType<MatType>::ReplicateType(const ReplicateType& other) :
    Layer<MatType>(other),
    multiples(other.multiples),
    sizeMult(other.sizeMult),
    outIdxs(other.outIdxs),
    coefs(other.coefs)
{
  // Nothing else to do.
}

template<typename MatType>
ReplicateType<MatType>::ReplicateType(ReplicateType&& other) :
    Layer<MatType>(std::move(other)),
    multiples(std::move(other.multiples)),
    sizeMult(other.sizeMult),
    outIdxs(std::move(other.outIdxs)),
    coefs(std::move(other.coefs))
{
  // Nothing else to do.
}

template<typename MatType>
ReplicateType<MatType>&
ReplicateType<MatType>::operator=(const ReplicateType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    multiples = other.multiples;
    sizeMult = other.sizeMult;
    outIdxs = other.outIdxs;
    coefs = other.coefs;
  }

  return *this;
}

template<typename MatType>
ReplicateType<MatType>& ReplicateType<MatType>::operator=(ReplicateType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    multiples = std::move(other.multiples);
    sizeMult = other.sizeMult;
    outIdxs = std::move(other.outIdxs);
    coefs = std::move(other.coefs);
  }

  return *this;
}

template<typename MatType>
void ReplicateType<MatType>::Forward(const MatType& input, MatType& output)
{
  // since the tensors are flattened to columns, we are just multiplying n_rows
  // by n for the total outputs
  output.set_size(input.n_rows * sizeMult, input.n_cols);

  output = input.rows(outIdxs);
}

template<typename MatType>
void ReplicateType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g.set_size(input.n_rows, input.n_cols);

  g = coefs * gy;
}

template<typename MatType>
template<typename Archive>
void ReplicateType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(multiples));
  ar(CEREAL_NVP(sizeMult));
  ar(CEREAL_NVP(outIdxs));
  ar(CEREAL_NVP(coefs));
}

} // namespace mlpack

#endif
