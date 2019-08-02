/**
 * @file batch_norm_impl.hpp
 * @author Praveen Ch
 * @author Manthan-R-Sheth
 *
 * Implementation of the Batch Normalization Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_BATCHNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_BATCHNORM_IMPL_HPP

// In case it is not included.
#include "batch_norm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm() :
    size(0),
    eps(1e-8),
    loading(false),
    deterministic(false),
    count(0)
{
  // Nothing to do here.
}
template <typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm(
    const size_t size, const double eps) :
    size(size),
    eps(eps),
    loading(false),
    deterministic(false),
    count(0)
{
  weights.set_size(size + size, 1);
  runningMean.zeros(size, 1);
  runningVariance.zeros(size, 1);
}

template<typename InputDataType, typename OutputDataType>
void BatchNorm<InputDataType, OutputDataType>::Reset()
{
  gamma = arma::mat(weights.memptr(), size, 1, false, false);
  beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false);

  if (!loading)
  {
    gamma.fill(1.0);
    beta.fill(0.0);
  }

  deterministic = false;
  loading = false;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // Mean and variance over the entire training set will be used to compute
  // the forward pass when deterministic is set to true.
  if (deterministic)
  {
    // Normalize the input and scale and shift the output.
    output = input.each_col() - runningMean;
    output.each_col() %= gamma / arma::sqrt(runningVariance / count + eps);
    output.each_col() += beta;
  }
  else
  {
    mean = arma::mean(input, 1);
    variance = arma::var(input, 1, 1);

    // Normalize the input.
    output = input.each_col() - mean;
    inputMean = output;
    output.each_col() /= arma::sqrt(variance + eps);

    // Use Welford method to compute the sample variance and mean.
    for (size_t i = 0; i < input.n_cols; i++)
    {
      count += 1;

      OutputDataType diff = input.col(i) - runningMean;
      runningMean = runningMean + diff / count;
      runningVariance += diff % (input.col(i) - runningMean);
    }

    // Reused in the backward and gradient step.
    normalized = output;

    // Scale and shift the output.
    output.each_col() %= gamma;
    output.each_col() += beta;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);

  // Step 1: dl / dxhat.
  const arma::mat norm = gy.each_col() % gamma;

  // Step 2: sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  const arma::mat var = arma::sum(norm % inputMean, 1) %
      arma::pow(stdInv, 3.0) * -0.5;

  // Step 4: dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  g = (norm.each_col() % stdInv) + (inputMean.each_col() %
      var * 2 / input.n_cols);

  // Step 3: sum (dl / dxhat * -1 / stdInv) + variance *
  // (sum -2 * (x - mu)) / m.
  g.each_col() += arma::sum(norm.each_col() % -stdInv, 1) / input.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Gradient(
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
void BatchNorm<InputDataType, OutputDataType>::serialize(
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
  ar & BOOST_SERIALIZATION_NVP(count);
  ar & BOOST_SERIALIZATION_NVP(runningMean);
  ar & BOOST_SERIALIZATION_NVP(runningVariance);
}

} // namespace ann
} // namespace mlpack

#endif
