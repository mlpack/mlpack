/**
 * @file methods/ann/layer/concatenate_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Concatenate class that concatenates a constant matrix to
 * the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCATENATE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCATENATE_IMPL_HPP

// In case it hasn't yet been included.
#include "concatenate.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>::
ConcatenateType(const InputType& concat) :
  concat(concat)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>::
ConcatenateType(const ConcatenateType& layer) :
  concat(layer.concat)
{
  // Nothing to to here.
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>::
ConcatenateType(ConcatenateType&& layer) :
  concat(std::move(layer.concat))
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>&
ConcatenateType<InputType, OutputType>::
operator=(const ConcatenateType& layer)
{
  if (this != &layer)
  {
    concat = layer.concat;
  }

  return *this;
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>&
ConcatenateType<InputType, OutputType>::
operator=(ConcatenateType&& layer)
{
  if (this != &layer)
  {
    concat = std::move(layer.concat);
  }

  return *this;
}

template<typename InputType, typename OutputType>
void ConcatenateType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (concat.is_empty())
    Log::Warn << "Concatenate::Forward(): the concat matrix is empty or was "
        << "not provided." << std::endl;

  output = arma::join_rows(input,
      arma::repmat(arma::vectorise(concat), input.n_cols));
}

template<typename InputType, typename OutputType>
void ConcatenateType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  // TODO: was this change correct?
  g = gy.submat(0, 0, gy.n_rows - concat.n_elem, gy.n_cols - 1);
}

} // namespace ann
} // namespace mlpack

#endif
