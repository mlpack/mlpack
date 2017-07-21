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
#ifndef MLPACK_METHODS_ANN_LAYER_MEMORY_UNIT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEMORY_UNIT_IMPL_HPP

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
    const size_t memSize,
    const size_t shiftSize) :
    inSize(inSize),
    outSize(outSize),
    memSize(memSize),
    shiftSize(shiftSize),
    deterministic(false)
{
  readHead = new MemoryHead<>(inSize, outSize, memSize, shiftSize);

  //writeHead = new MemoryHead<>(inSize, outSize, memSize, shiftSize);

  network.push_back(readHead);
  //network.push_back(writeHead);

  //controllerToLinear = new Linear<>(outSize, 2 * memSize);
  //eraseGate = new SigmoidLayer<>();
  //addGate = new TanHLayer<>();

  //network.push_back(controllerToLinear);
  //network.push_back(eraseGate);
  //network.push_back(addGate);

  memoryHistory.push_back(arma::ones(outSize, memSize));
  bMemoryHistory = memoryHistory.end();

  //lReads.push_back(arma::ones(memSize, 1));

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
  //arma::vec newInput = arma::join_vert(input, lReads.back());

  // Forward pass through the controller.
  //boost::apply_visitor(ForwardVisitor(std::move(newInput), std::move(
  //    boost::apply_visitor(outputParameterVisitor, controller.front()))),
  //    controller.front());

  //for (size_t i = 1; i < controller.size(); ++i)
  //{
  //  boost::apply_visitor(ForwardVisitor(
  //      std::move(boost::apply_visitor(outputParameterVisitor, controller[i - 1])),
  //      std::move(boost::apply_visitor(outputParameterVisitor, controller[i]))),
  //      controller[i]);
  //}

  // Get controller output.
  //arma::mat& controllerOutput = boost::apply_visitor(outputParameterVisitor,
  //  network.back());

  // Acess to current memory.
  arma::mat& cMemory = memoryHistory.back();

  // Pass the controller output through read head.
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(input),
      std::move(cMemory),
      std::move(boost::apply_visitor(outputParameterVisitor, readHead))),
      readHead);
  const arma::mat& readWeights = boost::apply_visitor(outputParameterVisitor,
      readHead);

  // Read memory with read weights
  //lReads.push_back(readWeights * cMemory);

  // Pass the controller output through write head.
  //boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
  //    std::move(cMemory),
  //    boost::apply_visitor(outputParameterVisitor, writeHead)),
  //    writeHead);
  //const arma::mat& writeWeights = boost::apply_visitor(outputParameterVisitor,
  //    writeHead);

  // Pass the controller output through linear layer.
  //boost::apply_visitor(ForwardVisitor(std::move(controllerOutput), std::move(
  //    boost::apply_visitor(outputParameterVisitor, controllerToLinear))),
  //    controllerToLinear);

  // Generate erase vector.
  //boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(outputParameterVisitor, controllerToLinear).submat(0, 0, memSize - 1, 0)),
  //    std::move(boost::apply_visitor(outputParameterVisitor, eraseGate))),
  //    eraseGate);
  //const arma::mat& eraseVec = boost::apply_visitor(outputParameterVisitor, eraseGate);

  // Generate add vector.
  //boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(outputParameterVisitor, controllerToLinear).submat(memSize, 0, 2 * memSize - 1, 0)),
  //    std::move(boost::apply_visitor(outputParameterVisitor, addGate))),
  //    addGate);
  //const arma::mat& addVec = boost::apply_visitor(outputParameterVisitor, addGate);

  // Perform erase and add to memory.
  //arma::mat nMemory = cMemory;

  //auto writeWeightsIt = writeWeights.begin();

  //nMemory.each_row([&](arma::rowvec& v)
  //{
  //  auto eraseVecIt = eraseVec.begin();
  //  auto addVecIt = addVec.begin();

  //  v.for_each([&](double& val)
  //  {
  //    val = (val * (1 - (*eraseVecIt) * (*writeWeightsIt))) + ((*writeWeightsIt) * (*addVecIt));
  //    eraseVecIt++;
  //    addVecIt++;
  //  });

  //  writeWeightsIt++;
  //});

  // Store the new memory.
  memoryHistory.push_back(cMemory);

  output = readWeights;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if(bMemoryHistory == memoryHistory.end())
  {
    bMemoryHistory = --(--memoryHistory.end());
  }

  arma::mat& memory = *bMemoryHistory;

  prevError = gy;

  boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, readHead)), std::move(memory), std::move(gy),
      std::move(boost::apply_visitor(deltaVisitor, readHead))),
      readHead);

  g = boost::apply_visitor(deltaVisitor, readHead);

  bMemoryHistory--;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(prevError)),
      readHead);
}

template<typename InputDataType, typename OutputDataType>
void NeuralTuringMachine<InputDataType, OutputDataType>::ResetCell()
{
  memoryHistory.clear();

  memoryHistory.push_back(arma::ones(outSize, memSize));
  bMemoryHistory = memoryHistory.end();
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
