/**
 * @file methods/ann/layer/sum_reduce_impl.hpp
 * @author Andrew Furey
 *
 * Definition of the SumReduce class sums inputs along a given axis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_REDUCE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_REDUCE_IMPL_HPP

// In case it hasn't yet been included.
#include "sum_reduce.hpp"

namespace mlpack {

template<typename MatType>
SumReduce<MatType>::SumReduce(size_t axis, bool keepDimensions) :
    Layer<MatType>(),
    axis(axis),
    keepDimensions(keepDimensions)
{
  // Nothing to do.
}

template<typename MatType>
SumReduce<MatType>::SumReduce(const SumReduce& other) :
    Layer<MatType>(other),
    axis(other.axis),
    keepDimensions(other.keepDimensions)
{
  // Nothing to do.
}

template<typename MatType>
SumReduce<MatType>::SumReduce(SumReduce&& other) :
    Layer<MatType>(std::move(other)),
    axis(std::move(other.axis)),
    keepDimensions(std::move(other.keepDimensions))
{
  // Nothing to do.
}

template<typename MatType>
SumReduce<MatType>&
SumReduce<MatType>::operator=(const SumReduce& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    axis = other.axis;
    keepDimensions = other.keepDimensions;
  }

  return *this;
}

template<typename MatType>
SumReduce<MatType>&
SumReduce<MatType>::operator=(SumReduce&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    axis = std::move(other.axis);
    keepDimensions = std::move(other.keepDimensions);
  }

  return *this;
}

template<typename MatType>
void SumReduce<MatType>::Forward(const MatType& input, MatType& output)
{
  CubeType inputAlias;
  MakeAlias(inputAlias, input, rows, this->inputDimensions[axis],
    slices * input.n_cols);
  output = reshape((MatType)sum(inputAlias, 1),
    input.n_rows / this->inputDimensions[axis], input.n_cols);
}

template<typename MatType>
void SumReduce<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g.set_size(input.n_rows, input.n_cols);
  CubeType gAlias;
  MakeAlias(gAlias, g, rows, this->inputDimensions[axis],
    slices * input.n_cols);

  CubeType gyAlias;
  MakeAlias(gyAlias, gy, rows, 1, slices * gy.n_cols);

  for (size_t i = 0; i < this->inputDimensions[axis]; i++)
    gAlias.col(i) = gyAlias;
}

template<typename MatType>
void SumReduce<MatType>::ComputeOutputDimensions()
{
  if (axis >= this->inputDimensions.size())
  {
    std::ostringstream errMessage;
    errMessage << "SumReduce::ComputeOutputDimensions(): Cannot "
        << "sum along axis " << axis << " when there are "
        << this->inputDimensions.size() << " input dimensions.";
    throw std::logic_error(errMessage.str());
  }
  this->outputDimensions = this->inputDimensions;

  if (keepDimensions || this->outputDimensions.size() == 1)
    this->outputDimensions[axis] = 1;
  else
    this->outputDimensions.erase(this->outputDimensions.begin() + axis);

  rows = 1;
  for (size_t i = 0; i < axis; i++)
    rows *= this->inputDimensions[i];

  slices = 1;
  for (size_t i = axis + 1; i < this->inputDimensions.size(); i++)
    slices *= this->inputDimensions[i];
}

template<typename MatType>
template<typename Archive>
void SumReduce<MatType>::serialize(Archive& ar,
  const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
  ar(CEREAL_NVP(axis));
  ar(CEREAL_NVP(keepDimensions));
}

} // namespace mlpack

#endif
