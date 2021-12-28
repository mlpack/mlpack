/**
 * @file methods/ann/layer/isru_impl.hpp
 * @author Suvarsha Chennareddy
 *
 * Definition of the ISRU activation function as described by Jonathan T. Barron.
 *
 * For more information, read the following paper (page 6).
 *
 * @code
 * @article{
 *   author  = {Carlile, Brad and Delamarter, Guy and Kinney, Paul and Marti,
 *              Akiko and Whitney, Brian},
 *   title   = {Improving deep learning by inverse square root linear units (ISRLUs)},
 *   year    = {2017},
 *   url     = {https://arxiv.org/pdf/1710.09967.pdf}
 * }
 * @endcode
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

template<typename InputDataType, typename OutputDataType>
ISRU<InputDataType, OutputDataType>::ISRU(const double alpha):
    alpha(alpha)
{
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
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivative(arma::size(gy));
  
  derivative = arma::pow(1 / (arma::sqrt(1 + alpha * arma::pow(input, 2))), 3);

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ISRU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(alpha));
}

} // namespace ann
} // namespace mlpack

#endif
