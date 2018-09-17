/**
 * @file concatenate_impl.hpp
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

template<typename InputDataType, typename OutputDataType>
Concatenate<InputDataType, OutputDataType>::Concatenate()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Concatenate<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  if (concat.is_empty())
    Log::Warn << "The concat matrix has not been provided." << std::endl;

  if (input.n_cols != concat.n_cols)
  {
    Log::Fatal << "The number of columns of the concat matrix should be equal "
        << "to the number of columns of input matrix." << std::endl;
  }

  inRows = input.n_rows;
  output = arma::join_cols(input, concat);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Concatenate<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    const arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  g = gy.submat(0, 0, inRows - 1, concat.n_cols - 1);
}

} // namespace ann
} // namespace mlpack

#endif
