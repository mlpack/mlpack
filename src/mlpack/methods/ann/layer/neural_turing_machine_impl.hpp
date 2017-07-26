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

  readHead = new MemoryHead<>(outSize, numMem, memSize, shiftSize);
  writeHead = new MemoryHead<>(outSize, numMem, memSize, shiftSize);

  network.push_back(readHead);
  network.push_back(writeHead);

  //controllerToLinear = new Linear<>(outSize, 2 * memSize);
  //controllerToLinearError.set_size(2 * memSize, 1);

  //eraseGate = new SigmoidLayer<>();
  //addGate = new TanHLayer<>();

  //network.push_back(controllerToLinear);
  //network.push_back(eraseGate);
  //network.push_back(addGate);

  memoryHistory.push_back(arma::ones(numMem, memSize));
  bMemoryHistory = memoryHistory.end();

  lReads.push_back(arma::ones(memSize, 1));
  gReads = lReads.end();

  backwardStep = 0;

  dMem = arma::zeros(numMem, memSize);

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

  // Pass the controller output through read head.
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
      std::move(cMemory),
      std::move(boost::apply_visitor(outputParameterVisitor, readHead))),
      readHead);
  const arma::mat& readWeights = boost::apply_visitor(outputParameterVisitor,
      readHead);

  // Read memory with read weights
  lReads.push_back(arma::trans(cMemory) * readWeights);

  // Pass the controller output through write head.
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
      std::move(cMemory),
      std::move(boost::apply_visitor(outputParameterVisitor, writeHead))),
      writeHead);
  const arma::mat& writeWeights = boost::apply_visitor(outputParameterVisitor,
      writeHead);

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
  arma::mat nMemory = cMemory;

  arma::vec eraseVec = 0.2 * arma::ones(memSize, 1);
  arma::vec addVec = 0.2 * arma::vec(memSize, 1);

  auto eraseVecIt = eraseVec.begin();
  auto addVecIt = addVec.begin();

  nMemory.each_col([&](arma::vec& v)
  {
    v = (v - (*eraseVecIt * (writeWeights % v))) + (*addVecIt * writeWeights);

    eraseVecIt++;
    addVecIt++;
  });

  // Store the new memory.
  memoryHistory.push_back(nMemory);

  output = controllerOutput;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if(bMemoryHistory == memoryHistory.end())
  {
    bMemoryHistory = --(memoryHistory.end());
    backwardStep = 0;

    dMem = arma::zeros(numMem, memSize);
  }
  else
  {
    // Load the memory content used at this time.
    arma::mat& memory = *bMemoryHistory;

    arma::mat tempGrad;
    if (backwardStep > 1)
    {
      // pass gradient through AddGate.
      //arma::mat dAddGate = arma::trans(dMem) * boost::apply_visitor(
      //    outputParameterVisitor, writeHead);

     // boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
     //     outputParameterVisitor, addGate)), std::move(dAddGate),
     //     std::move(boost::apply_visitor(deltaVisitor, addGate))),
     //     addGate);

      //controllerToLinearError.submat(memSize, 0, 2 * memSize - 1, 0) = boost::apply_visitor(deltaVisitor, addGate);

      // Delta of writeWeighst.
      dWriteHead = dMem * (0.2 * arma::ones(memSize, 1));//boost::apply_visitor(outputParameterVisitor, addGate);
      dWriteHead += (memory % dMem) * (0.2 * arma::ones(memSize, 1));//boost::apply_visitor(outputParameterVisitor, eraseGate);

      // pass gradient through EraseGate.
      //arma::mat dEraseGate = arma::trans(memory % dMem) * boost::apply_visitor(
      //    outputParameterVisitor, writeHead);

      //boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      //    outputParameterVisitor, eraseGate)), std::move(dEraseGate),
      //    std::move(boost::apply_visitor(deltaVisitor, eraseGate))),
      //    eraseGate);

      //controllerToLinearError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor, eraseGate);

      // Backward pass through controllerToLinear.
      //boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      //    outputParameterVisitor, controllerToLinear)), std::move(controllerToLinearError),
      //    std::move(boost::apply_visitor(deltaVisitor, controllerToLinear))),
      //    controllerToLinear);

      // Add to the error of the output.
      //gy += boost::apply_visitor(deltaVisitor, controllerToLinear);

      auto writeWeightsIt = boost::apply_visitor(outputParameterVisitor, writeHead).begin();

      // Get gradient with respect to current memory.
      dMem.each_col([&] (arma::vec& v)
      {
        v -= (*writeWeightsIt) * (v % arma::trans(0.2 * arma::ones(memSize, 1)));  //boost::apply_visitor(outputParameterVisitor, eraseGate)));

        writeWeightsIt++;
      });

      // Backward pass through write head.
      boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, writeHead)), std::move(memory), std::move(dWriteHead),
        std::move(boost::apply_visitor(deltaVisitor, writeHead)), std::move(tempGrad)),
        writeHead);

      // Add the memory gradient from write head.
      dMem += tempGrad;

      // Add the gradient to write gate to output gradient.
      gy += boost::apply_visitor(deltaVisitor, writeHead);
    }

    // Delta of the read
    dRead = memory * boost::apply_visitor(deltaVisitor, controller.front()).submat(inSize, 0, inSize + memSize - 1, 0);

    // Add memory gradient from read operation.
    dMem += boost::apply_visitor(outputParameterVisitor, readHead) * arma::trans(boost::apply_visitor(deltaVisitor, controller.front()).submat(inSize, 0, inSize + memSize - 1, 0));

    // Backward delta of read.
    boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, readHead)), std::move(memory), std::move(dRead),
        std::move(boost::apply_visitor(deltaVisitor, readHead)), std::move(tempGrad)),
        readHead);

    // Add memory gradient from read head.
    dMem += tempGrad;

    // Add gradient from read head to output gradient.
    gy += boost::apply_visitor(deltaVisitor, readHead);
  }

  prevError = gy;

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

  // Return the delta of the input
  g = boost::apply_visitor(deltaVisitor, controller.front()).submat(0, 0, inSize - 1, 0);

  bMemoryHistory--;
  backwardStep++;
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
    gradientStep= 0;
  }
  else
  {
    if(gradientStep > 1)
    {
      // Gradient of the write head
      boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, controller.back())),
          std::move(dWriteHead)),
          writeHead);

      // Gradient of controllerToLinear
      //boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
      //    outputParameterVisitor, controller.back())),
      //    std::move(controllerToLinearError)),
      //    controllerToLinear);
    }

    // Gradient of the Read head
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, controller.back())),
        std::move(dRead)),
        readHead);
  }

  const arma::mat& read = *gReads;

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

  boost::apply_visitor(GradientVisitor(std::move(arma::join_vert(input, read)),
      std::move(boost::apply_visitor(deltaVisitor, controller[1]))),
      controller.front());

  gReads--;
  gradientStep++;
}

template<typename InputDataType, typename OutputDataType>
void NeuralTuringMachine<InputDataType, OutputDataType>::ResetCell()
{
  memoryHistory.clear();

  memoryHistory.push_back(arma::ones(numMem, memSize));
  bMemoryHistory = memoryHistory.end();

  lReads.clear();
  lReads.push_back(arma::ones(memSize, 1));
  gReads = lReads.end();

  backwardStep = 0;
  gradientStep = 0;

  dMem = arma::zeros(numMem, memSize);
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
