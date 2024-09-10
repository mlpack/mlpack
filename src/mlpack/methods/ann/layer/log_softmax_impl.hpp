/**
 * @file methods/ann/layer/log_softmax_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LogSoftmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_IMPL_HPP

// In case it hasn't yet been included.
#include "log_softmax.hpp"

namespace mlpack {

template<typename MatType>
LogSoftMaxType<MatType>::LogSoftMaxType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
LogSoftMaxType<MatType>::LogSoftMaxType(const LogSoftMaxType& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
LogSoftMaxType<MatType>::LogSoftMaxType(LogSoftMaxType&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
LogSoftMaxType<MatType>&
LogSoftMaxType<MatType>::operator=(const LogSoftMaxType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
LogSoftMaxType<MatType>&
LogSoftMaxType<MatType>::operator=(LogSoftMaxType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
  }

  return *this;
}

template<typename MatType>
void LogSoftMaxType<MatType>::Forward(const MatType& input, MatType& output)
{
  ForwardImpl(input, output);
}

template<typename MatType>
void LogSoftMaxType<MatType>::ForwardImpl(
    const MatType& input,
    MatType& output,
    const typename std::enable_if_t<arma::is_arma_type<MatType>::value>*)
{
  MatType maxInput = repmat(max(input, 0), input.n_rows, 1);
  output = (maxInput - input);

  // Approximation of the base-e exponential function. The accuracy, however, is
  // about 0.00001 lower than using exp. Credits go to Leon Bottou.
  #pragma omp parallel for
  for (size_t i = 0; i < output.n_elem; ++i)
  {
    double x = output(i);
    //! Fast approximation of exp(-x) for x positive.
    static constexpr double A0 = 1.0;
    static constexpr double A1 = 0.125;
    static constexpr double A2 = 0.0078125;
    static constexpr double A3 = 0.00032552083;
    static constexpr double A4 = 1.0172526e-5;

    if (x < 13.0)
    {
      double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
      y *= y;
      y *= y;
      y *= y;
      y = 1 / y;
      output(i) = y;
    }
    else
    {
      output(i) = 0.0;
    }
  }

  #pragma omp parallel for
  for (size_t col = 0; col < maxInput.n_cols; ++col)
  {
    double colSum = 0.0;
    for (size_t row = 0; row < output.n_rows; ++row)
    {
      colSum += output(row, col);
    }
    double logSum = std::log(colSum);
    for (size_t row = 0; row < maxInput.n_rows; ++row)
    {
      maxInput(row, col) += logSum;
    }
  }

  output = input - maxInput;
}

#ifdef MLPACK_HAS_COOT

template<typename MatType>
void LogSoftMaxType<MatType>::ForwardImpl(
    const MatType& input,
    MatType& output,
    const typename std::enable_if_t<coot::is_coot_type<MatType>::value>*)
{
  MatType maxInput = repmat(max(input), input.n_rows, 1);
  output = (maxInput - input);
  output = exp(output * -1);
  maxInput.each_row() += log(sum(output));
  output = input - maxInput;
}

#endif

template<typename MatType>
void LogSoftMaxType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  g = gy - exp(output) % repmat(sum(gy), output.n_rows, 1);
}

} // namespace mlpack

#endif
