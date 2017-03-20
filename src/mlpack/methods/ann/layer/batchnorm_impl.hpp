/**
 * @file batchnorm_impl.hpp
 * @author Praveen Ch
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
#include "batchnorm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm(
    const size_t size, const double eps = 0.001) : size(size), eps(eps)
{
  weights.set_size(size + size, 1);
}


template<typename InputDataType, typename OutputDataType>
void BatchNorm<InputDataType, OutputDataType>::Reset()
{
  gamma = arma::mat(weights.memptr(), size, 1, false, false);
  beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false);
  deterministic = false;
  gamma.fill(1.0);
  beta.fill(0.0);
  stats.reset();
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  output.reshape(input.n_rows, input.n_cols);

  // Mean and variance over the entire training set will be used to compute
  // the forward pass when deterministic is set to true.
  if (deterministic)
  {
    for (size_t i = 0; i < output.n_cols; i++)
    {
      output.col(i) = input.col(i) - stats.mean().col(i);
    }

    for (size_t i = 0; i < output.n_rows; i++)
    {
      output.row(i) *= arma::as_scalar(gamma.row(i));
      output.row(i) /= arma::as_scalar(arma::sqrt(stats.var(1).row(i) + eps));
      output.row(i) += arma::as_scalar(beta.row(i));
    }
    
  }
  else
  {
    mean = arma::mean(input, 1);
    variance = arma::var(input, 1, 1);
    
    for (size_t i = 0; i < output.n_cols; i++)
    {
      output.col(i) = input.col(i) - mean;
      stats(input.col(i));
    }

    for (size_t i = 0; i < output.n_rows; i++)
    {
      output.row(i) = (input.row(i) - arma::as_scalar(arma::mean(input.row(i), 1)));
      output.row(i) /= (arma::as_scalar(arma::sqrt(variance.row(i) + eps)));
      output.row(i) *= arma::as_scalar(gamma.row(i));
      output.row(i) += arma::as_scalar(beta.row(i));
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  size_t n = input.n_cols;

  g.reshape(input.n_rows, input.n_cols);

  for (size_t i = 0; i < input.n_rows; ++i)
  {
    mean = arma::mean(input.row(i), 1);
    variance = arma::var(input.row(i), 1, 1);

    g.row(i) = -(input.row(i) - arma::as_scalar(mean));
    g.row(i) *= arma::as_scalar(arma::sum(gy.row(i) % 
        (input.row(i) - arma::as_scalar(mean)), 1));
    g.row(i) /= (arma::as_scalar(variance + eps));
    g.row(i) += (n * gy.row(i) - arma::as_scalar(arma::sum(gy.row(i),1)));
    g.row(i) *= (1.0 / n) * arma::as_scalar(gamma.row(i));
    g.row(i) /= (arma::as_scalar(arma::sqrt(variance + eps)));

  }

}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  arma::mat normalized(input.n_rows, input.n_cols);
  gradient.reshape(size + size, 1);
  
  for (size_t i = 0; i < normalized.n_rows; i++)
  {
    normalized.row(i) = (input.row(i) - arma::as_scalar(arma::mean(input.row(i), 1)));
    normalized.row(i) /= (arma::as_scalar(arma::sqrt(variance.row(i) + eps)));
  }

  gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized % error, 1);
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = arma::sum(error, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BatchNorm<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(gamma, "gamma");
  ar & data::CreateNVP(beta, "beta");
  ar & data::CreateNVP(stats.mean(), "trainingMean");
  ar & data::CreateNVP(stats.var(1), "trainingVariance");
}

} // namespace ann
} // namespace mlpack

#endif
