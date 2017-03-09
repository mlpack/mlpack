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


#include "batchnorm.hpp"

namespace mlpack {
namespace ann {

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
  // if(!deterministic)
  // {
  //   mean = arma::mean(input, 1);
  //   variance = arma::var(input, 1, 1);
  // }

  // output = beta + (gamma % (input - mean)) / arma::sqrt(variance + eps);
  output.reshape(input.n_rows, input.n_cols);

  for (size_t i = 0; i < output.n_rows; i++)
  {
    arma::mat inpRow = input.row(i);

    output.row(i) = (inpRow - arma::as_scalar(arma::mean(inpRow,1)));

    output.row(i) /= (arma::as_scalar(arma::sqrt(arma::var(inpRow,1, 1)) + 
        arma::as_scalar(eps)));

    output.row(i) *= arma::as_scalar(gamma.row(i));

    output.row(i) += arma::as_scalar(beta.row(i));

  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  size_t n = input.n_cols;

  g = (1/n) * gamma % arma::pow(variance + eps, -0.5) %
      (n * gy - arma::sum(gy, 1) - (input - mean) % arma::pow(variance + eps, -1.0) % arma::sum(gy % (input - mean), 1));
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum((input - mean) % arma::pow(variance + eps, -0.5) % error, 1);
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
