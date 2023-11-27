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
    const size_t _n, const size_t axis) :
    Layer<MatType>(),
    axis(axis),
    n(_n)
{
  // Nothing to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType() :
    Layer<MatType>(),
    axis(0),
    n(1)
{
  // Nothing to do.
}

template<typename MatType>
RepeatType<MatType>::~RepeatType()
{
  // Nothing to do: the child layer memory is already cleared by MultiLayer.
}

template<typename MatType>
RepeatType<MatType>::RepeatType(const RepeatType& other) :
    Layer<MatType>(other),
    axis(other.axis),
    n(other.n)
{
  // Nothing else to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType(RepeatType&& other) :
    Layer<MatType>(std::move(other)),
    axis(std::move(other.axis)),
    n(other.n)
{
  // Nothing else to do.
}

template<typename MatType>
RepeatType<MatType>& RepeatType<MatType>::operator=(const RepeatType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    axis = other.axis;
    n = other.n;
  }

  return *this;
}

template<typename MatType>
RepeatType<MatType>& RepeatType<MatType>::operator=(RepeatType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    axis = std::move(other.axis);
    n = other.n;
  }

  return *this;
}

template<typename MatType>
void RepeatType<MatType>::Forward(const MatType& input, MatType& output)
{
  // since the tensors are flattened to columns, we are just multiplying n_rows by n for the total outputs
  output.set_size(input.n_rows * n, input.n_cols);

  // Now alias the matrix so we can repelem it properly
  MatType inalias, outalias;
  MakeAlias(inalias, (typename MatType::elem_type*) input.memptr(), aliasRows, aliasCols * input.n_cols);
  MakeAlias(outalias, (typename MatType::elem_type*) output.memptr(), aliasRows * this->n, aliasCols * input.n_cols);
  if (axis == 0) {
    outalias = arma::repelem(inalias, n, 1);
  }
  else {
    outalias = arma::repmat(inalias, n, 1);
  }
}

template<typename MatType>
void RepeatType<MatType>::Backward(
    const MatType& input, const MatType& /* output */, const MatType& gy, MatType& g)
{
  g.set_size(input.n_rows, input.n_cols);

  // Now alias the matrix so we can repmat it properly
  MatType galias, gyalias;
  MakeAlias(galias, (typename MatType::elem_type*) g.memptr(), aliasRows, aliasCols * gy.n_cols);
  MakeAlias(gyalias, (typename MatType::elem_type*) gy.memptr(), aliasRows * this->n, aliasCols * gy.n_cols);

  if (axis == 0) {
    for (size_t i=0; i<galias.n_rows; i++) {
      galias.row(i) = arma::mean(gyalias.rows(i * aliasRows, (i+1) * aliasRows - 1), 0);
    }
  }
  else {
    galias = gyalias.rows(0, aliasRows - 1);
    for (size_t i = 1; i < this->n; i++) {
      galias += gyalias.rows(i * aliasRows, (i + 1) * aliasRows - 1);
    }
    galias /= this->n;
  }
}

template<typename MatType>
template<typename Archive>
void RepeatType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(axis));
  ar(CEREAL_NVP(n));
  ar(CEREAL_NVP(aliasRows));
  ar(CEREAL_NVP(aliasCols));
}

} // namespace mlpack

#endif
