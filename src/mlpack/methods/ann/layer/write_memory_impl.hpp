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
#include "../visitor/reset_cell_visitor.hpp"
#include "../visitor/forward_with_memory_visitor.hpp"
#include "../visitor/backward_with_memory_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
WriteMemory<InputDataType, OutputDataType>::WriteMemory(const size_t inSize,
                                                        const size_t numMem,
                                                        const size_t memSize,
                                                        const size_t shiftSize) :
  inSize(inSize),
  numMem(numMem),
  memSize(memSize),
  shiftSize(shiftSize)
{
  inputToLinear = new Linear<>(inSize, 2 * memSize);

  addGate = new TanHLayer<>();
  eraseGate = new SigmoidLayer<>();

  writeHead = new MemoryHead<>(inSize, numMem, memSize, shiftSize);

  network.push_back(inputToLinear);
  network.push_back(addGate);
  network.push_back(eraseGate);
  network.push_back(writeHead);

  dWriteHead.set_size(numMem, 1);
  prevError.set_size(2 * memSize, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void WriteMemory<InputDataType, OutputDataType>::ForwardWithMemory(
    arma::Mat<eT>&& input, const arma::Mat<eT>&& memory, arma::Mat<eT>&& output)
{
  // Pass through linear layer.
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, inputToLinear))),
      inputToLinear);

  arma::mat& lOutput = boost::apply_visitor(outputParameterVisitor, inputToLinear);

  // Generate AddVec
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(0, 0, memSize - 1, 0)), std::move(
      boost::apply_visitor(outputParameterVisitor, addGate))),
      addGate);
  const arma::mat& addVec = boost::apply_visitor(outputParameterVisitor, addGate);

  // Generate EraseVec
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(memSize, 0, 2 * memSize - 1, 0)), std::move(
      boost::apply_visitor(outputParameterVisitor, eraseGate))),
      eraseGate);
  const arma::mat& eraseVec = boost::apply_visitor(outputParameterVisitor, eraseGate);

  // Generate write weights
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(input),
      std::move(memory),
      std::move(boost::apply_visitor(outputParameterVisitor, writeHead))),
      writeHead);
  const arma::mat& writeWeights = boost::apply_visitor(outputParameterVisitor,
      writeHead);

  if(writeWeights.n_rows != memory.n_rows)
  {
    std::cout << "Incorrect Size" << std::endl;
  }

  output = memory;

  auto addVecIt = addVec.begin();
  auto eraseVecIt = eraseVec.begin();

  output.each_col([&](arma::vec& v)
  {
    v = (v - (*eraseVecIt * (writeWeights % v))) + (*addVecIt * writeWeights);

    addVecIt++;
    eraseVecIt++;
  });
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void WriteMemory<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* output */,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy, arma::Mat<eT>&& g, arma::Mat<eT>&& gM)
{
  const arma::mat& writeWeights = boost::apply_visitor(outputParameterVisitor, writeHead);

  // Backward through AddGate
  arma::mat dGate = arma::trans(gy) * boost::apply_visitor(outputParameterVisitor,
      writeHead);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, addGate)), std::move(dGate),
      std::move(boost::apply_visitor(deltaVisitor, addGate))),
      addGate);

  prevError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor, addGate);

  // Backward through EraseGate
  dGate = -(arma::trans(gy % memory) * writeWeights);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, eraseGate)), std::move(dGate),
      std::move(boost::apply_visitor(deltaVisitor, eraseGate))),
      eraseGate);

  prevError.submat(memSize, 0, 2 * memSize - 1, 0) = boost::apply_visitor(deltaVisitor, eraseGate);

  // Backward through linear layer.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputToLinear)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, inputToLinear))),
      inputToLinear);

  const arma::mat& addVec = boost::apply_visitor(outputParameterVisitor, addGate);
  const arma::mat& eraseVec = boost::apply_visitor(outputParameterVisitor, eraseGate);

  // Error of writeHead.
  size_t rowIndex = 0;
  gy.each_row([&] (arma::rowvec& v)
  {
    dWriteHead(rowIndex, 0) = arma::as_scalar(v * addVec - ((memory.row(rowIndex) % v) * eraseVec));

    rowIndex++;
  });

  // Backward through writeHead
  boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, writeHead)), std::move(memory), std::move(dWriteHead),
        std::move(boost::apply_visitor(deltaVisitor, writeHead)), std::move(gM)),
        writeHead);

  // Memory gradient from operations.
  rowIndex = 0;
  gy.each_row([&] (arma::rowvec& v)
  {
    v -= v % (arma::trans(eraseVec) * writeWeights(rowIndex, 0));

    rowIndex++;
  });
  gM += gy;

  // Error of input
  g = boost::apply_visitor(deltaVisitor, writeHead);
  g += boost::apply_visitor(deltaVisitor, inputToLinear);
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

  boost::apply_visitor(GradientVisitor(std::move(input),
          std::move(dWriteHead)),
          writeHead);
}

template<typename InputDataType, typename OutputDataType>
void WriteMemory<InputDataType, OutputDataType>::ResetCell()
{
  for(auto layer : network)
  {
    boost::apply_visitor(ResetCellVisitor(), layer);
  }
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
