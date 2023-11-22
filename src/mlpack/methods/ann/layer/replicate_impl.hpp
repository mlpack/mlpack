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
    const size_t _n, const size_t axis) :
    Layer<MatType>(),
    axis(axis),
    n(_n),
    useAxis(true)
{
  // Nothing to do.
}

template<typename MatType>
ReplicateType<MatType>::ReplicateType() :
    Layer<MatType>(),
    axis(0),
    n(1),
    useAxis(false)
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
    axis(other.axis),
    n(other.n),
    useAxis(other.useAxis)
{
  // Nothing else to do.
}

template<typename MatType>
ReplicateType<MatType>::ReplicateType(ReplicateType&& other) :
    Layer<MatType>(std::move(other)),
    axis(std::move(other.axis)),
    n(other.n),
    useAxis(std::move(other.useAxis))
{
  // Nothing else to do.
}

template<typename MatType>
ReplicateType<MatType>& ReplicateType<MatType>::operator=(const ReplicateType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    axis = other.axis;
    n = other.n;
    useAxis = other.useAxis;
  }

  return *this;
}

template<typename MatType>
ReplicateType<MatType>& ReplicateType<MatType>::operator=(ReplicateType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    axis = std::move(other.axis);
    n = other.n;
    useAxis = std::move(other.useAxis);
  }

  return *this;
}

template<typename MatType>
void ReplicateType<MatType>::Forward(const MatType& input, MatType& output)
{
  // since the tensors are flattened to columns, we are just multiplying n_rows by n for the total outputs
  output.set_size(input.n_rows * n, input.n_cols);

  // Now alias the matrix so we can repmat it properly
  MatType inalias((typename MatType::elem_type*) input.memptr(), aliasRows, aliasCols * input.n_cols, false, true);
  MatType outalias((typename MatType::elem_type*) output.memptr(), aliasRows * this->n, aliasCols * input.n_cols, false, true);

  // now, we need to re-shape
  outalias = arma::repmat(inalias, n, 1);
}

template<typename MatType>
void ReplicateType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  g.set_size(input.n_rows, input.n_cols);

  // Now alias the matrix so we can repmat it properly
  MatType galias((typename MatType::elem_type*) g.memptr(), aliasRows, aliasCols * gy.n_cols, false, true);
  MatType gyalias((typename MatType::elem_type*) gy.memptr(), aliasRows * this->n, aliasCols * gy.n_cols, false, true);

  galias = gyalias.rows(0, aliasRows-1);
  for (size_t i=1; i<this->n; i++) {
    galias += gyalias.rows(i * aliasRows, (i+1) * aliasRows - 1);
  }
  galias /= this->n;
}

template<typename MatType>
template<typename Archive>
void ReplicateType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(axis));
  ar(CEREAL_NVP(n));
  ar(CEREAL_NVP(useAxis));
}

} // namespace mlpack

#endif
