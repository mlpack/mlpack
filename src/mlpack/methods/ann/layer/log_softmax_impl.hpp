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
  MatType maxInput = repmat(max(input), input.n_rows, 1);
  output = (maxInput - input);

  // Function to calculate Padé approximant for exp(-x) for x positive
  auto padeApproximant = [](double x) {
    static constexpr double numCoeffs[] = {120, -60, 12}; 
    static constexpr double denCoeffs[] = {120, 60, 12}; 

    double num = numCoeffs[0] + x * (numCoeffs[1] + x * numCoeffs[2]);
    double den = denCoeffs[0] + x * (denCoeffs[1] + x * denCoeffs[2]);

    return num / den;
  };

  output.transform([padeApproximant](double x) {
    return padeApproximant(x);
  });

  maxInput.each_row() += log(sum(output));
  output = input - maxInput;
}

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
