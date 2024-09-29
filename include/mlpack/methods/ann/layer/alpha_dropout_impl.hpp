/**
 * @file methods/ann/layer/alpha_dropout_impl.hpp
 * @author Dakshit Agrawal
 *
 * Definition of the Alpha-Dropout class, which implements a regularizer that
 * randomly sets units to alpha-dash to prevent them from co-adapting and
 * makes an affine transformation so as to keep the mean and variance of
 * outputs at their original values.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_IMPL_HPP

// In case it hasn't yet been included.
#include "alpha_dropout.hpp"

namespace mlpack {

template<typename MatType>
AlphaDropoutType<MatType>::AlphaDropoutType(
    const double ratio,
    const double alphaDash) :
    Layer<MatType>(),
    ratio(ratio),
    alphaDash(alphaDash)
{
  Ratio(ratio);
}

template<typename MatType>
AlphaDropoutType<MatType>::AlphaDropoutType(const AlphaDropoutType& other) :
    Layer<MatType>(other),
    mask(other.mask),
    ratio(other.ratio),
    alphaDash(other.alphaDash),
    a(other.a),
    b(other.b)
{
  // Nothing to do.
}

template<typename MatType>
AlphaDropoutType<MatType>::AlphaDropoutType(AlphaDropoutType&& other) :
    Layer<MatType>(std::move(other)),
    mask(std::move(other.mask)),
    ratio(std::move(other.ratio)),
    alphaDash(std::move(other.alphaDash)),
    a(std::move(other.a)),
    b(std::move(other.b))
{
  // Nothing to do.
}

template<typename MatType>
AlphaDropoutType<MatType>&
AlphaDropoutType<MatType>::operator=(const AlphaDropoutType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    mask = other.mask;
    ratio = other.ratio;
    alphaDash = other.alphaDash;
    a = other.a;
    b = other.b;
  }

  return *this;
}

template<typename MatType>
AlphaDropoutType<MatType>&
AlphaDropoutType<MatType>::operator=(AlphaDropoutType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    mask = std::move(other.mask);
    ratio = std::move(other.ratio);
    alphaDash = std::move(other.alphaDash);
    a = std::move(other.a);
    b = std::move(other.b);
  }

  return *this;
}

template<typename MatType>
void AlphaDropoutType<MatType>::Forward(const MatType& input, MatType& output)
{
  // The dropout mask will not be multiplied during testing.
  if (!this->training)
  {
    output = input;
  }
  else
  {
    // Set values to alphaDash with probability ratio.  Then apply affine
    // transformation so as to keep mean and variance of outputs to their
    // original values.
    mask.randu(input.n_rows, input.n_cols);
    mask.transform( [&](double val) { return (val > ratio); } );
    output = (input % mask + alphaDash * (1 - mask)) * a + b;
  }
}

template<typename MatType>
void AlphaDropoutType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy % mask * a;
}

template<typename MatType>
template<typename Archive>
void AlphaDropoutType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(ratio));
  ar(CEREAL_NVP(alphaDash));
  ar(CEREAL_NVP(a));
  ar(CEREAL_NVP(b));

  // No need to serialize the mask, since it will be recomputed on the next
  // forward pass.  But we should clear it if we are loading.
  if (Archive::is_loading::value)
    mask.clear();
}

} // namespace mlpack

#endif
