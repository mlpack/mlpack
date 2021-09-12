/**
 * @file methods/ann/layer/sqnl_impl.hpp
 * @author Shaikh Yusuf Niaz
 *
 * Implementation of Square NonLinearity (SQNL) function as described by
 * Wuraola, Adedamola and Patel, Nitish.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SQNL_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SQNL_IMPL_HPP

// In case it hasn't yet been included.
#include "sqnl.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SQNL<InputDataType, OutputDataType>::SQNL() 
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void SQNL<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = arma::ones<OutputDataType>(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  { 
    if (0 <= input(i) && input(i) <= 2.0)
      output(i) = input(i) - std::pow(input(i),2)/4.0;
    else if (-2.0 <= input(i) && input(i) < 0)
      output(i) = input(i) + std::pow(input(i),2)/4.0;
    else if (input(i) < -2.0)
      output(i) = -1.0;
    else {};
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void SQNL<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  derivative.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (2.0 < input(i) || input(i) < -2.0)
	    derivative(i) = 0;
    else if (0 <= input(i) && input(i) <= 2.0)
	    derivative(i) = 1 - input(i)/2.0;
    else 
	    derivative(i) = 1 + input(i)/2.0;
  }
  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SQNL<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
