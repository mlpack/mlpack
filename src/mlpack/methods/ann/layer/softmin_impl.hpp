/**
 * @file methods/ann/layer/softmin_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Softmin class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMIN_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMIN_IMPL_HPP

// In case it hasn't yet been included.
#include "softmin.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Softmin<InputDataType, OutputDataType>::Softmin()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Softmin<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    OutputType& output)
{ 
  InputType inputMin = arma::repmat(arma::min(input,0), input.n_rows, 1);
  output = arma::repmat(arma::log(arma::sum(
      arma::exp(-(input - inputMin)),0)), input.n_rows, 1);
  output = arma::exp(-(input - inputMin) - output);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Softmin<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  //g = Need to figure out. 
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Softmin<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}
                                             
} // namespace ann
} // namespace mlpack

#endif
