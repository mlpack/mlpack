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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType, typename RegularizerType>
Linear3DType<InputType, OutputType, RegularizerType>::Linear3DType() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType, typename RegularizerType>
Linear3DType<InputType, OutputType, RegularizerType>::Linear3DType(
    const size_t inSize,
    const size_t outSize,
    RegularizerType regularizer) :
    inSize(inSize),
    outSize(outSize),
    regularizer(regularizer)
{
  weights.set_size(outSize * inSize + outSize, 1);
}

template<typename InputType, typename OutputType, typename RegularizerType>
Linear3DType<InputType, OutputType, RegularizerType>::Linear3DType(
    const Linear3DType& layer) :
    inSize(layer.inSize),
    outSize(layer.outSize),
    weights(layer.weights),
    regularizer(layer.regularizer)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType, typename RegularizerType>
Linear3DType<InputType, OutputType, RegularizerType>::Linear3DType(
    Linear3DType&& layer) :
    inSize(0),
    outSize(0),
    weights(std::move(layer.weights)),
    regularizer(std::move(layer.regularizer))
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType, typename RegularizerType>
Linear3DType<InputType, OutputType, RegularizerType>&
Linear3DType<InputType, OutputType, RegularizerType>::
operator=(const Linear3DType& layer)
{
  if (this != &layer)
  {
    inSize = layer.inSize;
    outSize = layer.outSize;
    weights = layer.weights;
    regularizer = layer.regularizer;
  }
  return *this;
}

template<typename InputType, typename OutputType, typename RegularizerType>
Linear3DType<InputType, OutputType, RegularizerType>&
Linear3DType<InputType, OutputType, RegularizerType>::
operator=(Linear3DType&& layer)
{
  if (this != &layer)
  {
    inSize = 0;
    outSize = 0;
    weights = std::move(layer.weights);
    regularizer = std::move(layer.regularizer);
  }
  return *this;
}

template<typename InputType, typename OutputType, typename RegularizerType>
void Linear3DType<InputType, OutputType, RegularizerType>::Reset()
{
  weight = OutputType(weights.memptr(), outSize, inSize, false, false);
  bias = OutputType(weights.memptr() + weight.n_elem, outSize, 1, false, false);
}

template<typename InputType, typename OutputType, typename RegularizerType>
void Linear3DType<InputType, OutputType, RegularizerType>::Forward(
    const InputType& input, OutputType& output)
{
  typedef typename arma::Cube<typename InputType::elem_type> CubeType;

  if (input.n_rows % inSize != 0)
  {
    Log::Fatal << "Number of features in the input must be divisible by inSize."
               << std::endl;
  }

  const size_t nPoints = input.n_rows / inSize;
  const size_t batchSize = input.n_cols;

  output.set_size(outSize * nPoints, batchSize);

  const CubeType inputTemp(const_cast<InputType&>(input).memptr(), inSize,
      nPoints, batchSize, false, false);

  for (size_t i = 0; i < batchSize; ++i)
  {
    // Shape of weight : (outSize, inSize).
    // Shape of inputTemp : (inSize, nPoints, batchSize).
    InputType z = weight * inputTemp.slice(i);
    z.each_col() += bias;
    output.col(i) = arma::vectorise(z);
  }
}

template<typename InputType, typename OutputType, typename RegularizerType>
void Linear3DType<InputType, OutputType, RegularizerType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  typedef typename arma::Cube<typename InputType::elem_type> CubeType;

  if (gy.n_rows % outSize != 0)
  {
    Log::Fatal << "Number of rows in propagated error must be divisible by \
        outSize." << std::endl;
  }

  const size_t nPoints = gy.n_rows / outSize;
  const size_t batchSize = gy.n_cols;

  const CubeType gyTemp(const_cast<InputType&>(gy).memptr(), outSize,
      nPoints, batchSize, false, false);

  g.set_size(inSize * nPoints, batchSize);

  for (size_t i = 0; i < gyTemp.n_slices; ++i)
  {
    // Shape of weight : (outSize, inSize).
    // Shape of gyTemp : (outSize, nPoints, batchSize).
    g.col(i) = arma::vectorise(weight.t() * gyTemp.slice(i));
  }
}

template<typename InputType, typename OutputType, typename RegularizerType>
void Linear3DType<InputType, OutputType, RegularizerType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  typedef typename arma::Cube<typename InputType::elem_type> CubeType;

  if (error.n_rows % outSize != 0)
    Log::Fatal << "Propagated error matrix has invalid dimension!" << std::endl;

  const size_t nPoints = input.n_rows / inSize;
  const size_t batchSize = input.n_cols;

  const CubeType inputTemp(const_cast<InputType&>(input).memptr(), inSize,
      nPoints, batchSize, false, false);
  const CubeType errorTemp(const_cast<InputType&>(error).memptr(), outSize,
      nPoints, batchSize, false, false);

  CubeType dW(outSize, inSize, batchSize);
  for (size_t i = 0; i < batchSize; ++i)
  {
    // Shape of errorTemp : (outSize, nPoints, batchSize).
    // Shape of inputTemp : (inSize, nPoints, batchSize).
    dW.slice(i) = errorTemp.slice(i) * inputTemp.slice(i).t();
  }

  gradient.set_size(arma::size(weights));

  gradient.submat(0, 0, weight.n_elem - 1, 0)
      = arma::vectorise(arma::sum(dW, 2));

  gradient.submat(weight.n_elem, 0, weights.n_elem - 1, 0)
      = arma::vectorise(arma::sum(arma::sum(errorTemp, 2), 1));

  regularizer.Evaluate(weights, gradient);
}

template<typename InputType, typename OutputType, typename RegularizerType>
template<typename Archive>
void Linear3DType<InputType, OutputType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (cereal::is_loading<Archive>())
    weights.set_size(outSize * inSize + outSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
