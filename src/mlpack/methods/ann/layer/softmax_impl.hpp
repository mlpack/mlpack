/**
 * @file methods/ann/layer/softmax_impl.hpp
 * @author Mrityunjay Tripathi
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
Softmax<InputDataType, OutputDataType>::Softmax()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Softmax<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    OutputType& output)
{
  InputType softmaxInput = arma::exp(input.each_row() -
      arma::max(input, 0));
  output = softmaxInput.each_row() / sum(softmaxInput, 0);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Softmax<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = input % (gy - arma::repmat(arma::sum(gy % input), input.n_rows, 1));
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Softmax<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
