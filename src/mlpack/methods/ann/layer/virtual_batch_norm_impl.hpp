/**
 * @file virtual_batch_norm_impl.hpp
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

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
VirtualBatchNorm<InputDataType, OutputDataType>::VirtualBatchNorm() :
    size(0),
    eps(1e-8),
    loading(false),
    oldCoefficient(0),
    newCoefficient(0)
{
  // Nothing to do here.
}
template <typename InputDataType, typename OutputDataType>
template<typename eT>
VirtualBatchNorm<InputDataType, OutputDataType>::VirtualBatchNorm(
    const arma::Mat<eT>& referenceBatch,
    const size_t size,
    const double eps) :
    size(size),
    eps(eps),
    loading(false)
{
  weights.set_size(size + size, 1);

  referenceBatchMean = arma::mean(referenceBatch, 1);
  referenceBatchMeanSquared = arma::mean(arma::square(referenceBatch), 1);
  newCoefficient = 1.0 / (referenceBatch.n_cols + 1);
  oldCoefficient = 1 - newCoefficient;
}

template<typename InputDataType, typename OutputDataType>
void VirtualBatchNorm<InputDataType, OutputDataType>::Reset()
{
  gamma = arma::mat(weights.memptr(), size, 1, false, false);
  beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false);

  if (!loading)
  {
    gamma.fill(1.0);
    beta.fill(0.0);
  }

  loading = false;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void VirtualBatchNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  inputParameter = input;
  arma::mat inputMean = arma::mean(input, 1);
  arma::mat inputMeanSquared = arma::mean(arma::square(input), 1);

  mean = oldCoefficient * referenceBatchMean + newCoefficient * inputMean;
  arma::mat meanSquared = oldCoefficient * referenceBatchMeanSquared +
      newCoefficient * inputMeanSquared;
  variance = meanSquared - arma::square(mean);
  // Normalize the input.
  output = input.each_col() - mean;
  inputSubMean = output;
  output.each_col() /= arma::sqrt(variance + eps);

  // Reused in the backward and gradient step.
  normalized = output;
  // Scale and shift the output.
  output.each_col() %= gamma;
  output.each_col() += beta;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void VirtualBatchNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);

  // dl / dxhat.
  const arma::mat norm = gy.each_col() % gamma;

  // sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  const arma::mat var = arma::sum(norm % inputSubMean, 1) %
      arma::pow(stdInv, 3.0) * -0.5;

  // dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * newCoefficient / m.
  g = (norm.each_col() % stdInv) + ((inputParameter.each_col() %
      var) * 2 * newCoefficient / inputParameter.n_cols);

  // (sum (dl / dxhat * -1 / stdInv) + (variance * mean * -2)) *
  // newCoefficient / m.
  g.each_col() += (arma::sum(norm.each_col() % -stdInv, 1) + (var %
      mean * -2)) * newCoefficient / inputParameter.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void VirtualBatchNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  gradient.set_size(size + size, 1);

  // Step 5: dl / dy * xhat.
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized % error, 1);

  // Step 6: dl / dy.
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(error, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void VirtualBatchNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(size);

  if (Archive::is_loading::value)
  {
    weights.set_size(size + size, 1);
    loading = false;
  }

  ar & BOOST_SERIALIZATION_NVP(eps);
  ar & BOOST_SERIALIZATION_NVP(gamma);
  ar & BOOST_SERIALIZATION_NVP(beta);
}

} // namespace ann
} // namespace mlpack

#endif
