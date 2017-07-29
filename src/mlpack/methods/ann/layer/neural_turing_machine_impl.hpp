/**
 * @file memory_unit_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LSTM class, which implements a lstm network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NTM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_NTM_IMPL_HPP

// In case it hasn't yet been included.
#include "neural_turing_machine_impl.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/forward_with_memory_visitor.hpp"
#include "../visitor/backward_with_memory_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
NeuralTuringMachine<InputDataType, OutputDataType>::NeuralTuringMachine(
    const size_t inSize,
    const size_t outSize,
    const size_t numMem,
    const size_t memSize,
    const size_t shiftSize) :
    inSize(inSize),
    outSize(outSize),
    numMem(numMem),
    memSize(memSize),
    shiftSize(shiftSize),
    deterministic(false)
{
  temp = new Linear<>(inSize + memSize, outSize);
  temp2 = new Linear<>(outSize, outSize);

  network.push_back(temp);
  network.push_back(temp2);

  controller.push_back(temp);
  controller.push_back(temp2);

  readMem = new ReadMemory<>(outSize, numMem, memSize, shiftSize);
  //writeMem = new WriteMemory(outSize, numMem, memSize, shiftSize);

  network.push_back(readMem);
  //network.push_back(writeMem);

  memoryHistory.push_back(arma::ones(numMem, memSize));
  bMemoryHistory = memoryHistory.end();

  lReads.push_back(arma::zeros(memSize, 1));
  gReads = lReads.end();

  //backwardStep = 0;

  //dMem = arma::zeros(numMem, memSize);

  // controller network
  // TODO: Create API for user to build the controller.
  //LayerTypes temp = new Linear<>(inSize + memSize, outSize);
  //network.push_back(temp);
  //controller.push_back(temp);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // Create input to the controller.
  arma::vec newInput = arma::join_vert(input, lReads.back());

  // Forward pass through the controller.
  boost::apply_visitor(ForwardVisitor(std::move(newInput), std::move(
      boost::apply_visitor(outputParameterVisitor, controller.front()))),
      controller.front());

  for (size_t i = 1; i < controller.size(); ++i)
  {
    boost::apply_visitor(ForwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, controller[i - 1])),
        std::move(boost::apply_visitor(outputParameterVisitor, controller[i]))),
        controller[i]);
  }

  // Get controller output.
  arma::mat& controllerOutput = boost::apply_visitor(outputParameterVisitor,
      controller.back());

  // Acess to current memory.
  arma::mat& cMemory = memoryHistory.back();

  // Pass the controller output through read memory layer.
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
      std::move(cMemory),
      std::move(boost::apply_visitor(outputParameterVisitor, readMem))),
      readMem);
  lReads.push_back(boost::apply_visitor(outputParameterVisitor, readMem));

  // Pass the controller output through write memory.
  //boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
  //    std::move(cMemory),
  //    std::move(boost::apply_visitor(outputParameterVisitor, writeMem))),
  //    writeMem);
  //memoryHistory.push_back(boost::apply_visitor(outputParameterVisitor, writeMem));

  memoryHistory.push_back(cMemory);

  output = controllerOutput;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if(bMemoryHistory == memoryHistory.end())
  {
    bMemoryHistory = --(--(memoryHistory.end()));
    backwardStep = 0;

    prevError = gy;
  }
  else
  {
    // Backward pass through read operation.
    boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, readMem)), std::move(*bMemoryHistory), std::move(dRead),
        std::move(boost::apply_visitor(deltaVisitor, readMem)), std::move(dMem)),
        readMem);

    prevError = gy + boost::apply_visitor(deltaVisitor, readMem);
  }

  // Backward through controller.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, controller.back())), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, controller.back()))),
      controller.back());

  for(int i = controller.size() - 2;i >= 0;i--)
  {
    boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, controller[i])),
        std::move(boost::apply_visitor(deltaVisitor, controller[i + 1])),
        std::move(boost::apply_visitor(deltaVisitor, controller[i]))),
        controller[i]);
  }

  // Set the error for last read.
  dRead = boost::apply_visitor(deltaVisitor, controller.front()).submat(inSize, 0, inSize + memSize - 1, 0);

  // Return the delta of the input
  g = boost::apply_visitor(deltaVisitor, controller.front()).submat(0, 0, inSize - 1, 0);

  bMemoryHistory--;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  if(gReads == lReads.end())
  {
    gReads = --(--lReads.end());
  }

  boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, controller.back())),
      std::move(dRead)),
      readMem);

  // Gradient of the controller
  boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, controller[controller.size() - 2])),
      std::move(prevError)), controller.back());

  for(size_t i = controller.size() - 2;i > 0;i--)
  {
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, controller[i - 1])),
        std::move(boost::apply_visitor(deltaVisitor, controller[i + 1]))),
        controller[i]);
  }

  boost::apply_visitor(GradientVisitor(std::move(arma::join_vert(input, *gReads)),
      std::move(boost::apply_visitor(deltaVisitor, controller[1]))),
      controller.front());

  gReads--;
}

template<typename InputDataType, typename OutputDataType>
void NeuralTuringMachine<InputDataType, OutputDataType>::ResetCell()
{
  memoryHistory.clear();
  memoryHistory.push_back(arma::ones(numMem, memSize));
  bMemoryHistory = memoryHistory.end();

  lReads.clear();
  lReads.push_back(arma::zeros(memSize, 1));
  gReads = lReads.end();
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void NeuralTuringMachine<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{

}

} // namespace ann
} // namespace mlpack

#endif
