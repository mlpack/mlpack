/**
 * @file methods/ann/layer/c_relu_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of CReLU layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_C_RELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_C_RELU_IMPL_HPP

// In case it hasn't yet been included.
#include "c_relu.hpp"

namespace mlpack {

template<typename MatType>
CReLUType<MatType>::CReLUType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
CReLUType<MatType>::CReLUType(
    const CReLUType& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
CReLUType<MatType>::CReLUType(
    CReLUType&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
CReLUType<MatType>&
CReLUType<MatType>::operator=(const CReLUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
CReLUType<MatType>&
CReLUType<MatType>::operator=(CReLUType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
  }

  return *this;
}

template<typename MatType>
void CReLUType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  typename MatType::elem_type zero = 0.0;
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_cols; ++i)
  {
    for (size_t j = 0; j < (size_t) input.n_rows; ++j)
    {
      output(j, i) = std::max(input(j, i), zero);
      output(j + input.n_rows, i) = std::max(-input(j, i), zero);
    }
  }
}

template<typename MatType>
void CReLUType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // While it doesn't really matter if what we choose for the gradient at
  // exactly 0, since we have the discontinuity from -1 to 1 between 0- and 0+,
  // it seems cleaner to set the gradient at x=0 to 0 instead of picking either
  // +1 or -1.
  g = gy.rows(0, input.n_rows - 1) % (input > 0) -
      gy.rows(input.n_rows, input.n_rows * 2 - 1) % (input < 0);
}

template<typename MatType>
void CReLUType<MatType>::ComputeOutputDimensions()
{
  // The CReLU gives twice as many outputs as inputs.
  // We treat this as flattening the input, and then doubling the number of
  // dimensions.
  this->outputDimensions.clear();
  if (this->inputDimensions.size() == 0)
    return;

  size_t totalDimensions = 1;
  for (size_t i = 0; i < this->inputDimensions.size(); ++i)
    totalDimensions *= this->inputDimensions[i];

  // Now double the number of dimensions.
  totalDimensions *= 2;

  this->outputDimensions.push_back(totalDimensions);
}

template<typename MatType>
template<typename Archive>
void CReLUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
}

} // namespace mlpack

#endif
