/**
 * @file memory_read_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of Read Memory layer, used in Neural Turing Machine.
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

#include "../visitor/reset_cell_visitor.hpp"
#include "../visitor/forward_with_memory_visitor.hpp"
#include "../visitor/backward_with_memory_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
ReadMemory<InputDataType, OutputDataType>::ReadMemory(const size_t inSize,
                                                      const size_t numMem,
                                                      const size_t memSize,
                                                      const size_t shiftSize) :
  inSize(inSize),
  numMem(numMem),
  memSize(memSize),
  shiftSize(shiftSize)
{
  readHead = new MemoryHead<>(inSize, numMem, memSize, shiftSize);

  network.push_back(readHead);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ReadMemory<InputDataType, OutputDataType>::ForwardWithMemory(
    arma::Mat<eT>&& input, const arma::Mat<eT>&& memory, arma::Mat<eT>&& output)
{
  // Forward pass through read head.
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(input),
      std::move(memory),
      std::move(boost::apply_visitor(outputParameterVisitor, readHead))),
      readHead);
  const arma::mat& readWeights = boost::apply_visitor(outputParameterVisitor,
      readHead);

  output = arma::trans(memory) * readWeights;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ReadMemory<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* output */,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy, arma::Mat<eT>&& g, arma::Mat<eT>&& gM)
{
  // Delta of the read weights.
  dReadHead = memory * gy;

  // Backward pass through readHead.
  boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, readHead)), std::move(memory),
        std::move(dReadHead), std::move(boost::apply_visitor(deltaVisitor,
        readHead)), std::move(gM)), readHead);

  g = boost::apply_visitor(deltaVisitor, readHead);

  // Delta of memory from read operation.
  gM += boost::apply_visitor(outputParameterVisitor,
      readHead) * arma::trans(gy);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void ReadMemory<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input),
          std::move(dReadHead)),
          readHead);
}

template<typename InputDataType, typename OutputDataType>
void ReadMemory<InputDataType, OutputDataType>::ResetCell()
{
  for (auto layer : network)
  {
    boost::apply_visitor(ResetCellVisitor(), layer);
  }
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
