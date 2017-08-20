/**
 * @file neural_turing_machine_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of NeuralTuringMachine layer.
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
    const size_t shiftSize,
    LayerTypes controller) :
    inSize(inSize),
    outSize(outSize),
    numMem(numMem),
    memSize(memSize),
    shiftSize(shiftSize),
    controller(controller),
    deterministic(false)
{
  network.push_back(controller);

  readHead = new MemoryHead<>(outSize, numMem, memSize, shiftSize);
  writeHead = new MemoryHead<>(outSize, numMem, memSize, shiftSize);

  network.push_back(readHead);
  network.push_back(writeHead);

  inputToLinear = new Linear<>(outSize, 2 * memSize);

  addGate = new TanHLayer<>();
  eraseGate = new SigmoidLayer<>();

  network.push_back(inputToLinear);
  network.push_back(addGate);
  network.push_back(eraseGate);
  network.push_back(writeHead);

  dWriteHead.set_size(numMem, 1);
  linearError.set_size(2 * memSize, 1);

  allZeros = arma::zeros<arma::mat>(memSize, 1);

  allOnes = arma::ones<arma::mat>(numMem, memSize);

  memoryHistory.push_back(std::move(arma::mat(allOnes.memptr(),
      allOnes.n_rows, allOnes.n_cols, false, true)));

  bMemoryHistory = memoryHistory.end();

  lReads.push_back(std::move(arma::mat(allZeros.memptr(),
      allZeros.n_rows, allZeros.n_cols, false, true)));

  gReads = lReads.end();

  forwardStep = 0;
  backwardStep = 0;
  gradientStep = 0;
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
      boost::apply_visitor(outputParameterVisitor, controller))),
      controller);

  // Get controller output.
  arma::mat& controllerOutput = boost::apply_visitor(outputParameterVisitor,
      controller);

  // Acess to current memory.
  arma::mat& cMemory = memoryHistory.back();

  // Pass the controller output through read head.
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
      std::move(cMemory),
      std::move(boost::apply_visitor(outputParameterVisitor, readHead))),
      readHead);
  const arma::mat& readWeights = boost::apply_visitor(outputParameterVisitor,
      readHead);

  //! Read memory.
  lReads.push_back(arma::trans(cMemory) * readWeights);

  // Pass through linear layer.
  boost::apply_visitor(ForwardVisitor(std::move(controllerOutput), std::move(
      boost::apply_visitor(outputParameterVisitor, inputToLinear))),
      inputToLinear);

  arma::mat& lOutput = boost::apply_visitor(outputParameterVisitor,
      inputToLinear);

  // Generate AddVec
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(0, 0,
      memSize - 1, 0)), std::move(boost::apply_visitor(outputParameterVisitor,
      addGate))), addGate);
  const arma::mat& addVec = boost::apply_visitor(outputParameterVisitor,
      addGate);

  // Generate EraseVec
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(memSize, 0,
      2 * memSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, eraseGate))), eraseGate);
  const arma::mat& eraseVec = boost::apply_visitor(outputParameterVisitor,
      eraseGate);

  // Generate write weights
  boost::apply_visitor(ForwardWithMemoryVisitor(std::move(controllerOutput),
      std::move(cMemory),
      std::move(boost::apply_visitor(outputParameterVisitor, writeHead))),
      writeHead);
  const arma::mat& writeWeights = boost::apply_visitor(outputParameterVisitor,
      writeHead);

  arma::mat nMemory = cMemory;

  auto addVecIt = addVec.begin();
  auto eraseVecIt = eraseVec.begin();

  nMemory.each_col([&](arma::vec& v)
  {
    v = (v - (*eraseVecIt * (writeWeights % v))) + (*addVecIt * writeWeights);

    addVecIt++;
    eraseVecIt++;
  });


  memoryHistory.push_back(std::move(nMemory));

  output = controllerOutput;

  forwardStep++;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void NeuralTuringMachine<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if (bMemoryHistory == memoryHistory.end())
  {
    bMemoryHistory = --(--(memoryHistory.end()));
    backwardStep = 0;

    prevError = gy;
  }
  else
  {
    if (backwardStep > 1)
    {
      arma::mat dMemPrev = std::move(dMem);

      // Delta of the read weights.
      dReadHead = (*bMemoryHistory) * boost::apply_visitor(deltaVisitor,
          controller).submat(inSize, 0, inSize + memSize - 1, 0);

      // Backward pass through read head.
      boost::apply_visitor(BackwardWithMemoryVisitor(std::move(
          boost::apply_visitor(outputParameterVisitor, readHead)),
          std::move(*bMemoryHistory), std::move(dReadHead),
          std::move(boost::apply_visitor(deltaVisitor, readHead)),
          std::move(dMem)), readHead);

      // Delta of memory from read operation.
      dMem += boost::apply_visitor(outputParameterVisitor,
          readHead) * arma::trans(boost::apply_visitor(deltaVisitor,
          controller).submat(inSize, 0, inSize + memSize - 1, 0));

      const arma::mat& writeWeights = boost::apply_visitor(outputParameterVisitor,
      writeHead);

      // Backward through AddGate
      arma::mat dGate = arma::trans(dMemPrev) * writeWeights;

      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, addGate)), std::move(dGate),
          std::move(boost::apply_visitor(deltaVisitor, addGate))),
          addGate);

      linearError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor,
          addGate);

      // Backward through EraseGate
      dGate = -(arma::trans(dMemPrev % *bMemoryHistory) * writeWeights);

      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, eraseGate)), std::move(dGate),
          std::move(boost::apply_visitor(deltaVisitor, eraseGate))),
          eraseGate);

      linearError.submat(memSize, 0, 2 * memSize - 1, 0) = boost::apply_visitor(
          deltaVisitor, eraseGate);

      // Backward through linear layer.
      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, inputToLinear)), std::move(linearError),
          std::move(boost::apply_visitor(deltaVisitor, inputToLinear))),
          inputToLinear);

      const arma::mat& addVec = boost::apply_visitor(outputParameterVisitor,
          addGate);
      const arma::mat& eraseVec = boost::apply_visitor(outputParameterVisitor,
          eraseGate);

      // Error of writeHead.
      size_t rowIndex = 0;
      dMemPrev.each_row([&] (const arma::rowvec& v)
      {
        dWriteHead(rowIndex, 0) = arma::as_scalar(v * addVec -
            (((*bMemoryHistory).row(rowIndex) % v) * eraseVec));

        rowIndex++;
      });

      // Backward through writeHead
      arma::mat dMemTemp;
      boost::apply_visitor(BackwardWithMemoryVisitor(std::move(boost::apply_visitor(
            outputParameterVisitor, writeHead)), std::move(*bMemoryHistory),
            std::move(dWriteHead), std::move(boost::apply_visitor(deltaVisitor,
            writeHead)), std::move(dMemTemp)), writeHead);
      dMem += dMemTemp;

      // Memory gradient from operations.
      rowIndex = 0;
      dMemPrev.each_row([&] (arma::rowvec& v)
      {
        v -= v % (arma::trans(eraseVec) * writeWeights(rowIndex, 0));

        rowIndex++;
      });
      dMem += dMemPrev;

      prevError = gy + boost::apply_visitor(deltaVisitor, readHead) +
          boost::apply_visitor(deltaVisitor, writeHead) +
          boost::apply_visitor(deltaVisitor, inputToLinear);
    }
    else
    {
      // Delta of the read weights.
      dReadHead = (*bMemoryHistory) * boost::apply_visitor(deltaVisitor,
          controller).submat(inSize, 0, inSize + memSize - 1, 0);

      // Backward pass through read head.
      boost::apply_visitor(BackwardWithMemoryVisitor(std::move(
          boost::apply_visitor(outputParameterVisitor, readHead)),
          std::move(*bMemoryHistory), std::move(dReadHead),
          std::move(boost::apply_visitor(deltaVisitor, readHead)),
          std::move(dMem)), readHead);

      // Delta of memory from read operation.
      dMem += boost::apply_visitor(outputParameterVisitor,
          readHead) * arma::trans(boost::apply_visitor(deltaVisitor,
          controller).submat(inSize, 0, inSize + memSize - 1, 0));

      prevError = gy + boost::apply_visitor(deltaVisitor, readHead);
    }
  }

  // Backward through controller.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, controller)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, controller))),
      controller);

  // Return the delta of the input
  g = boost::apply_visitor(deltaVisitor, controller).submat(0, 0,
      inSize - 1, 0);

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
  if (gReads == lReads.end())
  {
    gReads = --(--lReads.end());

    gradientStep = 0;
  }
  else
  {
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, controller)),
        std::move(dReadHead)),
        readHead);

    if (gradientStep > 1)
    {
      boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, controller)),
          std::move(linearError)),
          inputToLinear);

      boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, controller)),
          std::move(dWriteHead)),
          writeHead);
    }
  }

  // Gradient of the controller
  boost::apply_visitor(GradientVisitor(std::move(arma::join_vert(input,
      *gReads)), std::move(prevError)),
      controller);

  gReads--;
  gradientStep++;
}

template<typename InputDataType, typename OutputDataType>
void NeuralTuringMachine<InputDataType, OutputDataType>::ResetCell()
{
  memoryHistory.clear();

  memoryHistory.push_back(std::move(arma::mat(allOnes.memptr(),
      allOnes.n_rows, allOnes.n_cols, false, true)));

  bMemoryHistory = memoryHistory.end();

  lReads.clear();

  lReads.push_back(std::move(arma::mat(allZeros.memptr(),
      allZeros.n_rows, allZeros.n_cols, false, true)));

  gReads = lReads.end();

  backwardStep = 0;
  gradientStep = 0;
  forwardStep = 0;
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
