/**
 * @file methods/ann/layer/group_norm_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the Group Normalization class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_GROUPNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GROUPNORM_IMPL_HPP

// In case it is not included.
#include "group_norm.hpp"

namespace mlpack {


template<typename InputDataType, typename OutputDataType>
GroupNorm<InputDataType, OutputDataType>::GroupNorm() :
    groupCount(1),
    size(0),
    eps(1e-8),
    loading(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
GroupNorm<InputDataType, OutputDataType>::GroupNorm(
    const size_t groupCount, const size_t size, const double eps) :
    groupCount(groupCount),
    size(size),
    eps(eps),
    loading(false)
{
  if (size % groupCount != 0)
  {
    Log::Fatal << "Total input units must be divisible by groupCount!"
        << std::endl;
  }

  weights.set_size(size + size, 1);
}

template<typename InputDataType, typename OutputDataType>
void GroupNorm<InputDataType, OutputDataType>::Reset()
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
void GroupNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (output.is_empty())
    output.set_size(input.n_rows / groupCount, input.n_cols * groupCount);

  assert(size % groupCount == 0);
  assert(input.n_rows % size == 0);

  arma::mat reshapedInput(const_cast<arma::Mat<eT>&>(input).memptr(),
      input.n_rows / groupCount, input.n_cols * groupCount, false, false);

  mean = arma::mean(reshapedInput, 0);
  variance = arma::var(reshapedInput, 1, 0);

  // Normalize the input.
  output = reshapedInput.each_row() - mean;
  inputMean = output;
  output.each_row() /= sqrt(variance + eps);

  output.reshape(input.n_rows, input.n_cols);
  // Reused in the backward and gradient step.
  normalized = output;

  arma::mat expandedGamma, expandedBeta;
  expandedGamma.set_size(input.n_rows, 1);
  expandedBeta.set_size(input.n_rows, 1);

  for (size_t r = 0; r < input.n_rows; ++r)
  {
    expandedGamma(r) = gamma(r * size / input.n_rows);
    expandedBeta(r) = beta(r * size / input.n_rows);
  }

  // Scale and shift the output.
  output.each_col() %= expandedGamma;
  output.each_col() += expandedBeta;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  if (g.is_empty())
    g.set_size(input.n_rows / groupCount, input.n_cols * groupCount);

  arma::mat inputReshaped(const_cast<arma::Mat<eT>&>(input).memptr(),
      input.n_rows / groupCount, input.n_cols * groupCount, false, false);

  arma::mat expandedGamma;
  expandedGamma.set_size(input.n_rows, 1);
  for (size_t r = 0; r < input.n_rows; ++r)
  {
    expandedGamma(r) = gamma(r * size / input.n_rows);
  }

  // dl / dxhat.
  const arma::mat norm = gy.each_col() % expandedGamma;

  arma::mat normReshaped(const_cast<arma::Mat<eT>&>(norm).memptr(),
      gy.n_rows / groupCount, gy.n_cols * groupCount, false, false);

  const arma::mat stdInv = 1.0 / sqrt(variance + eps);

  // sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  const arma::mat var = sum(normReshaped % inputMean, 0) %
      pow(stdInv, 3.0) * -0.5;

  // dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  g = (normReshaped.each_row() % stdInv) + (inputMean.each_row() %
      var * 2 / inputReshaped.n_rows);

  // sum (dl / dxhat * -1 / stdInv) + variance *
  // (sum -2 * (x - mu)) / m.
  g.each_row() += sum(normReshaped.each_row() % -stdInv, 0) /
      inputReshaped.n_rows;
  g.reshape(input.n_rows, input.n_cols);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  assert(error.n_rows % size == 0);
  const size_t channelSize = error.n_rows / size;
  temp = sum(normalized % error, 1);
  arma::mat tempReshaped((temp).memptr(),
      channelSize, temp.n_elem / channelSize, false, false);

  gradient.set_size(size + size, 1);

  // Step 5: dl / dy * xhat.
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = sum(tempReshaped, 0).t();

  temp = sum(error, 1);
  arma::mat tempErrorReshaped((temp).memptr(),
      channelSize, temp.n_elem / channelSize, false, false);

  // Step 6: dl / dy.
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
      sum(tempErrorReshaped, 0).t();
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void GroupNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(groupCount));
  ar(CEREAL_NVP(size));

  if (cereal::is_loading<Archive>())
  {
    weights.set_size(size + size, 1);
    loading = true;
  }

  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(gamma));
  ar(CEREAL_NVP(beta));
}

} // namespace mlpack

#endif
