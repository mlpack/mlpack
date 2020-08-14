/**
 * @file methods/ann/layer/concat_performance_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the ConcatPerformance class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_IMPL_HPP

// In case it hasn't yet been included.
#include "concat_performance.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename OutputLayerType,
    typename InputDataType,
    typename OutputDataType
>
ConcatPerformance<
    OutputLayerType,
    InputDataType,
    OutputDataType
>::ConcatPerformance(const size_t inSize, OutputLayerType&& outputLayer) :
    inSize(inSize),
    outputLayer(std::move(outputLayer))
{
  // Nothing to do here.
}

template<
    typename OutputLayerType,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
double ConcatPerformance<
    OutputLayerType,
    InputDataType,
    OutputDataType
>::Forward(const arma::Mat<eT>& input, arma::Mat<eT>& target)
{
  const size_t elements = input.n_elem / inSize;

  double output = 0;
  for (size_t i = 0; i < input.n_elem; i+= elements)
  {
    arma::mat subInput = input.submat(i, 0, i + elements - 1, 0);
    output += outputLayer.Forward(subInput, target);
  }

  return output;
}

template<
    typename OutputLayerType,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void ConcatPerformance<
    OutputLayerType,
    InputDataType,
    OutputDataType
>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& target,
    arma::Mat<eT>& output)
{
  const size_t elements = input.n_elem / inSize;

  arma::mat subInput = input.submat(0, 0, elements - 1, 0);
  arma::mat subOutput;

  outputLayer.Backward(subInput, target, subOutput);

  output = arma::zeros(subOutput.n_elem, inSize);
  output.col(0) = subOutput;

  for (size_t i = elements, j = 0; i < input.n_elem; i+= elements, ++j)
  {
    subInput = input.submat(i, 0, i + elements - 1, 0);
    outputLayer.Backward(subInput, target, subOutput);

    output.col(j) = subOutput;
  }
}

template<
    typename OutputLayerType,
    typename InputDataType,
    typename OutputDataType
>
template<typename Archive>
void ConcatPerformance<
    OutputLayerType,
    InputDataType,
    OutputDataType
>::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(inSize);
}

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_performance_impl.hpp"

#endif
