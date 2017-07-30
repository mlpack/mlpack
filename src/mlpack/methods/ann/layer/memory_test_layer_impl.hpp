/**
 * @file memory_test_layer_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the Memory Test, which implements an abstraction layer to
 * test layers which handle memory.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEMORY_TEST_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEMORY_TEST_IMPL_HPP

// In case it hasn't yet been included.
#include "memory_test_layer.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/forward_with_memory_test_visitor.hpp"
#include "../visitor/backward_with_memory_test_visitor.hpp"
#include "../visitor/reset_cell_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
MemoryTest<InputDataType, OutputDataType>::MemoryTest(
    const size_t inSize,
    const size_t outSize,
    const size_t numMem,
    const size_t memSize,
    LayerTypes testLayer) :
    inSize(inSize),
    outSize(outSize),
    numMem(numMem),
    memSize(memSize),
    testLayer(testLayer),
    deterministic(false)
{
  initMem = new Linear<>(inSize, numMem * memSize);

  network.push_back(initMem);
  network.push_back(testLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryTest<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // Create memory content.
  boost::apply_visitor(ForwardVisitor(std::move(arma::ones(inSize, 1)),
      std::move(boost::apply_visitor(outputParameterVisitor, initMem))),
      initMem);

  arma::mat& memory = boost::apply_visitor(outputParameterVisitor, initMem);

  // Pass memory and input to the test layer.
  boost::apply_visitor(ForwardWithMemoryTestVisitor(std::move(input),
      std::move(arma::mat(memory.memptr(), numMem, memSize, false)),
      std::move(boost::apply_visitor(outputParameterVisitor, testLayer))),
      testLayer);

  output = boost::apply_visitor(outputParameterVisitor, testLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryTest<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* output */,
  arma::Mat<eT>&& gy,
  arma::Mat<eT>&& g)
{
  prevError = gy;

  arma::mat& memory = boost::apply_visitor(outputParameterVisitor, initMem);

  // Backward pass through testLayer.
  boost::apply_visitor(BackwardWithMemoryTestVisitor(std::move(
      boost::apply_visitor(outputParameterVisitor, testLayer)),
      std::move(arma::mat(memory.memptr(), numMem, memSize,
      false)), std::move(gy), std::move(boost::apply_visitor(deltaVisitor,
      testLayer)), std::move(dMem)), testLayer);

  g = boost::apply_visitor(deltaVisitor, testLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryTest<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input),
          std::move(prevError)),
          testLayer);

  boost::apply_visitor(GradientVisitor(std::move(arma::ones(inSize, 1)),
          std::move(arma::mat(dMem.memptr(), numMem * memSize, 1, false))),
          initMem);
}

template<typename InputDataType, typename OutputDataType>
void MemoryTest<InputDataType, OutputDataType>::ResetCell()
{
  boost::apply_visitor(ResetCellVisitor(), testLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MemoryTest<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  // Testing layer. Nothing to serialise
}

} // namespace ann
} // namespace mlpack

#endif
