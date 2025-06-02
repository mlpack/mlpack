/**
 * @file methods/ann/layer/virtual_batch_norm_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the VirtualBatchNorm layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_VIRTUALBATCHNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_VIRTUALBATCHNORM_IMPL_HPP

// In case it is not included.
#include "virtual_batch_norm.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename InputType, typename OutputType>
VirtualBatchNormType<InputType, OutputType>::VirtualBatchNormType() :
    size(0),
    eps(1e-8),
    loading(false),
    oldCoefficient(0),
    newCoefficient(0)
{
  // Nothing to do here.
}
template <typename InputType, typename OutputType>
VirtualBatchNormType<InputType, OutputType>::VirtualBatchNormType(
    const InputType& referenceBatch,
    const size_t size,
    const double eps) :
    size(size),
    eps(eps),
    loading(false)
{
  referenceBatchMean = arma::mean(referenceBatch, 1);
  referenceBatchMeanSquared = arma::mean(square(referenceBatch), 1);
  newCoefficient = 1.0 / (referenceBatch.n_cols + 1);
  oldCoefficient = 1 - newCoefficient;
}

template<typename InputType, typename OutputType>
void VirtualBatchNormType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  gamma = OutputType(weightsPtr, size, 1, false, false);
  beta = OutputType(weightsPtr + gamma.n_elem, size, 1, false, false);

  if (!loading)
  {
    gamma.fill(1.0);
    beta.fill(0.0);
  }

  loading = false;
}

template<typename InputType, typename OutputType>
void VirtualBatchNormType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  Log::Assert(input.n_rows % size == 0, "Input features must be divisible "
      "by feature maps.");

  inputParameter = input;
  InputType inputMean = arma::mean(input, 1);
  InputType inputMeanSquared = arma::mean(square(input), 1);

  mean = oldCoefficient * referenceBatchMean + newCoefficient * inputMean;
  OutputType meanSquared = oldCoefficient * referenceBatchMeanSquared +
      newCoefficient * inputMeanSquared;
  variance = meanSquared - square(mean);
  // Normalize the input.
  output = input.each_col() - mean;
  inputSubMean = output;
  output.each_col() /= sqrt(variance + eps);

  // Reused in the backward and gradient step.
  normalized = output;
  // Scale and shift the output.
  output.each_col() %= gamma;
  output.each_col() += beta;
}

template<typename InputType, typename OutputType>
void VirtualBatchNormType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  const OutputType stdInv = 1.0 / sqrt(variance + eps);

  // dl / dxhat.
  const OutputType norm = gy.each_col() % gamma;

  // sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  const OutputType var = sum(norm % inputSubMean, 1) % pow(stdInv, 3.0) * -0.5;

  // dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * newCoefficient / m.
  g = (norm.each_col() % stdInv) + ((inputParameter.each_col() %
      var) * 2 * newCoefficient / inputParameter.n_cols);

  // (sum (dl / dxhat * -1 / stdInv) + (variance * mean * -2)) *
  // newCoefficient / m.
  g.each_col() += (sum(norm.each_col() % -stdInv, 1) + (var %
      mean * -2)) * newCoefficient / inputParameter.n_cols;
}

template<typename InputType, typename OutputType>
void VirtualBatchNormType<InputType, OutputType>::Gradient(
    const InputType& /* input */,
    const OutputType& error,
    OutputType& gradient)
{
  gradient.set_size(size + size, 1);

  // Step 5: dl / dy * xhat.
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = sum(normalized % error, 1);

  // Step 6: dl / dy.
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = sum(error, 1);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void VirtualBatchNormType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(eps));

  if (cereal::is_loading<Archive>())
  {
    weights.set_size(size + size, 1);
    loading = true;
  }
}

} // namespace mlpack

#endif
