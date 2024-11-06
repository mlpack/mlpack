/**
 * @file methods/ann/layer/linear3d_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Linear layer class which accepts 3D input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR3D_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR3D_IMPL_HPP

// In case it hasn't yet been included.
#include "linear3d.hpp"

namespace mlpack {

template<typename MatType, typename RegularizerType>
Linear3DType<MatType, RegularizerType>::Linear3DType() :
    Layer<MatType>(),
    outSize(0)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
Linear3DType<MatType, RegularizerType>::Linear3DType(
    const size_t outSize,
    RegularizerType regularizer) :
    Layer<MatType>(),
    outSize(outSize),
    regularizer(regularizer)
{ }

template<typename MatType, typename RegularizerType>
Linear3DType<MatType, RegularizerType>::Linear3DType(
    const Linear3DType& other) :
    Layer<MatType>(other),
    outSize(other.outSize),
    regularizer(other.regularizer)
{
  // Nothing to do.
}

template<typename MatType, typename RegularizerType>
Linear3DType<MatType, RegularizerType>::Linear3DType(
    Linear3DType&& other) :
    Layer<MatType>(std::move(other)),
    outSize(std::move(other.outSize)),
    regularizer(std::move(other.regularizer))
{
  // Nothing to do.
}

template<typename MatType, typename RegularizerType>
Linear3DType<MatType, RegularizerType>&
Linear3DType<MatType, RegularizerType>::operator=(
    const Linear3DType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    outSize = other.outSize;
    regularizer = other.regularizer;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
Linear3DType<MatType, RegularizerType>&
Linear3DType<MatType, RegularizerType>::operator=(
    Linear3DType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    outSize = std::move(other.outSize);
    regularizer = std::move(other.regularizer);
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
void Linear3DType<MatType, RegularizerType>::SetWeights(
    const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, outSize * this->inputDimensions[0] + outSize,
      1);
  MakeAlias(weight, weightsIn, outSize, this->inputDimensions[0]);
  MakeAlias(bias, weightsIn, outSize, 1, weight.n_elem);
}

template<typename MatType, typename RegularizerType>
void Linear3DType<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
{
  using CubeType = arma::Cube<typename MatType::elem_type>;

  const size_t nPoints = input.n_rows / this->inputDimensions[0];
  const size_t batchSize = input.n_cols;

  const CubeType inputTemp(const_cast<MatType&>(input).memptr(),
      this->inputDimensions[0], nPoints, batchSize, false, false);

  for (size_t i = 0; i < batchSize; ++i)
  {
    // Shape of weight : (outSize, inSize).
    // Shape of inputTemp : (inSize, nPoints, batchSize).
    MatType z = weight * inputTemp.slice(i);
    z.each_col() += bias;
    output.col(i) = vectorise(z);
  }
}

template<typename MatType, typename RegularizerType>
void Linear3DType<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  using CubeType = arma::Cube<typename MatType::elem_type>;

  if (gy.n_rows % outSize != 0)
  {
    Log::Fatal << "Number of rows in propagated error must be divisible by "
        << "outSize." << std::endl;
  }

  const size_t nPoints = gy.n_rows / outSize;
  const size_t batchSize = gy.n_cols;

  const CubeType gyTemp(const_cast<MatType&>(gy).memptr(), outSize,
      nPoints, batchSize, false, false);

  for (size_t i = 0; i < gyTemp.n_slices; ++i)
  {
    // Shape of weight : (outSize, inSize).
    // Shape of gyTemp : (outSize, nPoints, batchSize).
    g.col(i) = vectorise(weight.t() * gyTemp.slice(i));
  }
}

template<typename MatType, typename RegularizerType>
void Linear3DType<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  using CubeType = arma::Cube<typename MatType::elem_type>;

  if (error.n_rows % outSize != 0)
    Log::Fatal << "Propagated error matrix has invalid dimension!" << std::endl;

  const size_t nPoints = input.n_rows / this->inputDimensions[0];
  const size_t batchSize = input.n_cols;

  const CubeType inputTemp(const_cast<MatType&>(input).memptr(),
      this->inputDimensions[0], nPoints, batchSize, false, false);
  const CubeType errorTemp(const_cast<MatType&>(error).memptr(), outSize,
      nPoints, batchSize, false, false);

  CubeType dW(outSize, this->inputDimensions[0], batchSize);
  for (size_t i = 0; i < batchSize; ++i)
  {
    // Shape of errorTemp : (outSize, nPoints, batchSize).
    // Shape of inputTemp : (inSize, nPoints, batchSize).
    dW.slice(i) = errorTemp.slice(i) * inputTemp.slice(i).t();
  }

  gradient.submat(0, 0, weight.n_elem - 1, 0) = vectorise(sum(dW, 2));

  gradient.submat(weight.n_elem, 0, weights.n_elem - 1, 0)
      = vectorise(sum(sum(errorTemp, 2), 1));

  regularizer.Evaluate(weights, gradient);
}

template<typename MatType, typename RegularizerType>
void Linear3DType<
    MatType, RegularizerType
>::ComputeOutputDimensions()
{
  // The Linear3D layer shares weights for each row of the input, and
  // duplicates it across the columns.  Thus, we only change the number of
  // rows.
  this->outputDimensions = this->inputDimensions;
  this->outputDimensions[0] = outSize;
}

template<typename MatType, typename RegularizerType>
template<typename Archive>
void Linear3DType<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
