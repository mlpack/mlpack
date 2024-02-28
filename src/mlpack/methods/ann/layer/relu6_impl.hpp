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

template<typename MatType>
ReLU6Type<MatType>::ReLU6Type() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
ReLU6Type<MatType>::ReLU6Type(
    const ReLU6Type& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
ReLU6Type<MatType>::ReLU6Type(
    ReLU6Type&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
ReLU6Type<MatType>&
ReLU6Type<MatType>::operator=(const ReLU6Type& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
ReLU6Type<MatType>&
ReLU6Type<MatType>::operator=(ReLU6Type&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
  }

  return *this;
}

template<typename MatType>
void ReLU6Type<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = arma::clamp(input, 0.0, 6.0);
}

template<typename MatType>
void ReLU6Type<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  #pragma omp for
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (input(i) < 6 && input(i) > 0)
      g(i) = gy(i);
    else
      g(i) = 0.0;
  }
}

template<typename MatType>
template<typename Archive>
void ReLU6Type<MatType>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace mlpack

#endif
