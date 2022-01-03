/**
 * @file methods/ann/layer/randomized_relu_impl.hpp
 * @author Shubham Agrawal
 *
 * @code
 * @article{Bing Xu2015,
 *  author = {Bing Xu, Naiyan Wang, Tianqi Chen, Mu Li},
 *  title = {Empirical Evaluation of Rectified Activations in Convolutional 
 *      Network},
 *  year = {2015},
 *  url = {https://arxiv.org/pdf/1505.00853.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RRELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RRELU_IMPL_HPP

// In case it hasn't yet been included.
#include "randomized_relu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
RReLU<InputDataType, OutputDataType>::RReLU(
    const double lowerbound,
    const double higherbound):
    lowerbound(lowerbound),
    higherbound(higherbound),
    deterministic(false),
    alpha(0.0)
{
  if (lowerbound < 1.0 || higherbound < 1.0)
  {
    Log::Fatal << "Lowerbound and higherbound must be greater than 1."
        << std::endl;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void RReLU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  if (deterministic)
  {
    alpha = 2.0 / (higherbound + lowerbound);
  }
  else
  {
    alpha = 1.0 / math::Random(lowerbound, higherbound);
  }
  output = arma::max(input, alpha * input);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void RReLU<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivative;
  derivative.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
    derivative(i) = (input(i) >= 0) ? 1 : alpha;

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void RReLU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(lowerbound));
  ar(CEREAL_NVP(higherbound));
}

} // namespace ann
} // namespace mlpack

#endif
