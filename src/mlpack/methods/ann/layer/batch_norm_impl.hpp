/**
 * @file batchnorm_impl.hpp
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
BatchNorm<InputDataType, OutputDataType>::BatchNorm()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm(
  const size_t size, const double eps) : size(size), eps(eps)
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
    mean = stats.mean();
    variance = stats.var(1);
  }
  else
  {
    mean = arma::mean(input, 1);
    variance = arma::var(input, 1, 1);

    for (size_t i = 0; i < output.n_cols; i++)
    {
      stats(input.col(i));
    }
  }

  output = input.each_col() - mean;
  output.each_col() %= gamma/arma::sqrt(variance+eps);
  output.each_col() += beta;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  size_t n = input.n_cols;

  g.reshape(input.n_rows, input.n_cols);

  mean = arma::mean(input, 1);
  variance = arma::var(input, 1, 1);

  arma::mat m = arma::sum(gy % (input.each_col() - mean), 1);
  g = arma::repmat(m, 1, input.n_cols) % -(input.each_col() - mean);
  g.each_col() %= 1.0/(variance + eps);
  g += (n * gy - arma::repmat((arma::sum(gy, 1)), 1, input.n_cols));
  g.each_col() %= ((1.0 / n) * gamma);
  g.each_col() %= (1.0/arma::sqrt(variance + eps));
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

  normalized = input.each_col() - arma::mean(input, 1)
                                  / arma::sqrt(arma::var(input, 1, 1) + eps);

  gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized % error, 1);
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
                                                          arma::sum(error, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BatchNorm<InputDataType, OutputDataType>::serialize(
  Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(gamma);
  ar & BOOST_SERIALIZATION_NVP(beta);
}

} // namespace ann
} // namespace mlpack

#endif
