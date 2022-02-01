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
ConcatenateType(const ConcatenateType& other) :
    Layer<InputType, OutputType>(other),
    concat(other.concat)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>::
ConcatenateType(ConcatenateType&& other) :
    Layer<InputType, OutputType>(std::move(other)),
    concat(other.concat)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>&
ConcatenateType<InputType, OutputType>::operator=(const ConcatenateType& other)
{
  if (&other != this)
  {
    Layer<InputType, OutputType>::operator=(other);
    concat = other.concat;
  }

  return *this;
}

template<typename InputType, typename OutputType>
ConcatenateType<InputType, OutputType>&
ConcatenateType<InputType, OutputType>::operator=(ConcatenateType&& other)
{
  if (&other != this)
  {
    Layer<InputType, OutputType>::operator=(std::move(other));
    concat = std::move(other.concat);
  }

  return *this;
}

template<typename InputType, typename OutputType>
void ConcatenateType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (concat.is_empty())
  {
    Log::Warn << "Concatenate::Forward(): the concat matrix is empty or was "
        << "not provided." << std::endl;
  }

  output.submat(0, 0, input.n_rows - 1, input.n_cols - 1) = input;
  output.submat(input.n_rows, 0, output.n_rows - 1, input.n_cols - 1) =
      arma::repmat(arma::vectorise(concat), 1, input.n_cols);
}

template<typename InputType, typename OutputType>
void ConcatenateType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  // Pass back the non-concatenated part.
  g = gy.submat(0, 0, gy.n_rows - 1 - concat.n_elem, gy.n_cols - 1);
}

} // namespace ann
} // namespace mlpack

#endif
