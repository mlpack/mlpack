/**
 * @file memory_unit_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of memory head layer, used in Neural Turing Machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_READ_MEMORY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_READ_MEMORY_IMPL_HPP

// In case it hasn't yet been included.
#include "read_memory.hpp"

#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
ReadMemory<InputDataType, OutputDataType>::ReadMemory()
{
  // Nothing to do.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ReadMemory<InputDataType, OutputDataType>::ForwardWithMemory(
    arma::Mat<eT>&& input, const arma::Mat<eT>&& memory, arma::Mat<eT>&& output)
{
  output = arma::trans(memory) * input;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ReadMemory<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* output */,
  const arma::Mat<eT>&& input,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy, arma::Mat<eT>&& g, arma::Mat<eT>&& gM)
{
  // Delta of the read.
  g = memory * gy;

  // Delta of memory.
  gM = input * arma::trans(gy);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ReadMemory<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{
}

} // namespace ann
} // namespace mlpack

#endif
