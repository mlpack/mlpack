/**
 * @file isru_impl.hpp
 * @author Prince Gupta
 *
 * Implementation of ISRU (Inverse Square Root Unit) activation function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ISRU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ISRU_IMPL_HPP

// In case it hasn't yet been included.
#include "isru.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

// This constructor is called for ISRU activation function.
// 'alpha' is a hyperparameter.
template<typename InputDataType, typename OutputDataType>
ISRU<InputDataType, OutputDataType>::ISRU(const double alpha) :
    alpha(alpha)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void ISRU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = input / (arma::sqrt(1 + alpha * arma::pow(input, 2)));
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void ISRU<InputDataType, OutputDataType>::Backward(
    const DataType& input, DataType& gy, DataType& g)
{
  double yEdge = 1 / std::sqrt(alpha);
  double x;
  g.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; i++)
  {
    if (input(i) == 0)
    {
      g(i) = 1;
      continue;
    }
    if (input(i) >= yEdge || input(i) <= -yEdge)
      x = std::signbit(input(i)) ? -DBL_MAX : DBL_MAX;
    else
      x = input(i) / std::sqrt(1 - alpha * std::pow(input(i), 2));

    g(i) = std::pow(input(i) / x, 3);
  }

  g = gy % g;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ISRU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(alpha);
}

} // namespace ann
} // namespace mlpack

#endif
