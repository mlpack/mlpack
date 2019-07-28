/**
 * @file softmax_impl.hpp
 * @author Sreenik Seal
 *
 * Implementation of the Softmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP

// In case it hasn't yet been included.
#include "softmax.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SoftMax<InputDataType, OutputDataType>::SoftMax()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void SoftMax<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  arma::mat maxInput = arma::repmat(arma::max(input), input.n_rows, 1);
  output = arma::exp(maxInput - input);

  // Approximation of the base-e exponential function. The acuracy however is
  // about 0.00001 lower as using exp. Credits go to Leon Bottou.
  output.transform([](double x)
  {
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

      return y;
    }

    return 0.0;
  });

  maxInput.each_row() += arma::sum(output));
  output = input / maxInput;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void SoftMax<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  int indexOfMax = input.index_max();
  double maxVal = input[indexOfMax];
  g[indexOfMax] = 1;
  g = g * maxVal - maxVal * input;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SoftMax<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
