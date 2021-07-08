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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
ReLu6<InputDataType, OutputDataType>::ReLu6()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void ReLu6<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output.zeros();
  output = arma::min(arma::max(output, input), 6.0);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void ReLu6<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{

}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ReLu6<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
