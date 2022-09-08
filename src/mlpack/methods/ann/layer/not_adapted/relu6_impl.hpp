/**
 * @file methods/ann/layer/relu6_impl.hpp
 * @author Aakash kaushik
 *
 * For more information, kindly refer to the following paper.
 *
 * @code
 * @article{Andrew G2017,
 *  author = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
 *      Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
 *  title = {MobileNets: Efficient Convolutional Neural Networks for Mobile
 *      Vision Applications},
 *  year = {2017},
 *  url = {https://arxiv.org/pdf/1704.04861}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RELU6_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RELU6_IMPL_HPP

// In case it hasn't yet been included.
#include "relu6.hpp"

namespace mlpack {

template<typename InputDataType, typename OutputDataType>
ReLU6<InputDataType, OutputDataType>::ReLU6()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void ReLU6<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{ 
  OutputType outputTemp(arma::size(input));
  outputTemp.fill(6.0);
  output = arma::zeros<OutputType>(arma::size(input));
  output = arma::min(arma::max(output, input), outputTemp);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void ReLU6<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivative(arma::size(gy));
  derivative.fill(0.0);
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (input(i) < 6 && input(i) > 0)
      derivative(i) = 1.0;
  }

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ReLU6<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace mlpack

#endif
