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
#ifndef MLPACK_METHODS_ANN_LAYER_WRITE_MEMORY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_WRITE_MEMORY_IMPL_HPP

// In case it hasn't yet been included.
#include "write_memory.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
WriteMemory<InputDataType, OutputDataType>::WriteMemory(const size_t inSize,
                                                        const size_t memSize) :
  inSize(inSize),
  memSize(memSize)
{
  inputToLinear = new Linear<>(inSize, memSize);

  addGate = new TanHLayer<>();

  network.push_back(inputToLinear);
  network.push_back(addGate);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void WriteMemory<InputDataType, OutputDataType>::ForwardWithMemory(
    arma::Mat<eT>&& input, const arma::Mat<eT>&& memory, arma::Mat<eT>&& output)
{
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, inputToLinear))),
      inputToLinear);

  boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(outputParameterVisitor, inputToLinear)), std::move(
      boost::apply_visitor(outputParameterVisitor, addGate))),
      addGate);

  output = memory;

  auto addVecIt = boost::apply_visitor(outputParameterVisitor, addGate).begin();

  output.each_col([&](arma::vec& v)
  {
    v += (*addVecIt * arma::ones(memory.n_rows, 1));

    addVecIt++;
  });
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void WriteMemory<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* output */,
  const arma::Mat<eT>&& /* input */,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy, arma::Mat<eT>&& g, arma::Mat<eT>&& gM)
{
  arma::mat dAddGate = arma::trans(gy) * arma::ones(memory.n_rows, 1);

  gM = gy;

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, addGate)), std::move(dAddGate),
      std::move(boost::apply_visitor(deltaVisitor, addGate))),
      addGate);

  prevError = boost::apply_visitor(deltaVisitor, addGate);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputToLinear)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, inputToLinear))),
      inputToLinear);

  g = boost::apply_visitor(deltaVisitor, inputToLinear);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void WriteMemory<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input),
          std::move(prevError)),
          inputToLinear);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void WriteMemory<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{
}

} // namespace ann
} // namespace mlpack

#endif
