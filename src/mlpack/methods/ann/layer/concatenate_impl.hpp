/**
 * @file methods/ann/layer/concatenate_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Concatenate class that concatenates a constant matrix to
 * the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCATENATE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCATENATE_IMPL_HPP

// In case it hasn't yet been included.
#include "concatenate.hpp"

namespace mlpack {

template<typename MatType>
ConcatenateType<MatType>::
ConcatenateType(const MatType& concat) :
    Layer<MatType>(),
    concat(concat)
{
  // Nothing to do here.
}

template<typename MatType>
ConcatenateType<MatType>::
ConcatenateType(const ConcatenateType& other) :
    Layer<MatType>(other),
    concat(other.concat)
{
  // Nothing to do.
}

template<typename MatType>
ConcatenateType<MatType>::
ConcatenateType(ConcatenateType&& other) :
    Layer<MatType>(std::move(other)),
    concat(other.concat)
{
  // Nothing to do.
}

template<typename MatType>
ConcatenateType<MatType>&
ConcatenateType<MatType>::operator=(const ConcatenateType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    concat = other.concat;
  }

  return *this;
}

template<typename MatType>
ConcatenateType<MatType>&
ConcatenateType<MatType>::operator=(ConcatenateType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    concat = std::move(other.concat);
  }

  return *this;
}

template<typename MatType>
void ConcatenateType<MatType>::Forward(const MatType& input, MatType& output)
{
  if (concat.is_empty())
  {
    Log::Warn << "Concatenate::Forward(): the concat matrix is empty or was "
        << "not provided." << std::endl;
  }

  output.submat(0, 0, input.n_rows - 1, input.n_cols - 1) = input;
  output.submat(input.n_rows, 0, output.n_rows - 1, input.n_cols - 1) =
      repmat(vectorise(concat), 1, input.n_cols);
}

template<typename MatType>
void ConcatenateType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // Pass back the non-concatenated part.
  g = gy.submat(0, 0, gy.n_rows - 1 - concat.n_elem, gy.n_cols - 1);
}

template<typename MatType>
void ConcatenateType<MatType>::ComputeOutputDimensions()
{
  // This flattens the input.
  size_t inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
      inSize *= this->inputDimensions[i];

  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
      1);
  this->outputDimensions[0] = inSize + concat.n_elem;
}

/**
 * Serialize the layer.
 */
template<typename MatType>
template<typename Archive>
void ConcatenateType<MatType>::serialize(
  Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(concat));
}

} // namespace mlpack

#endif
