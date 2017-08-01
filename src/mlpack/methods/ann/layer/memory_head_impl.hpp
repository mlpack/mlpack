/**
 * @file memory_unit_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of Memory Head layer, used in Neural Turing Machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEMORY_HEAD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEMORY_HEAD_IMPL_HPP

// In case it hasn't yet been included.
#include "memory_head.hpp"

#include <algorithm>

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
MemoryHead<InputDataType, OutputDataType>::MemoryHead(
    const size_t inSize,
    const size_t outSize,
    const size_t memSize,
    const size_t shiftRange) :
    inSize(inSize),
    outSize(outSize),
    memSize(memSize),
    shiftSize(shiftRange),
    deterministic(false)
{
  // Build linear for kT + bT + gT + sT + gammaT
  inputLinear = new Linear<>(inSize, (memSize));

  // kT non linearity.
  kTNonLinear = new TanHLayer<>();

  network.push_back(inputLinear);
  network.push_back(kTNonLinear);

  prevWeights.push_back(arma::zeros<arma::mat>(outSize, 1));
  weightsBackwardIterator = prevWeights.end();

  prevError = arma::zeros<arma::mat>((memSize), 1);

  bWdash = lWDash.end();
  bGammaT = lGammaT.end();
  bWTilde = lWTilde.end();
  bShiftMatrix = lShiftMatrix.end();
  bWg = lWg.end();
  bSt = lSt.end();
  bGt = lGt.end();
  bWe = lWe.end();
  bWc = lWc.end();
  bBt = lBt.end();
  bCosineT = lConsineT.end();
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::ForwardWithMemory(
    arma::Mat<eT>&& input, const arma::Mat<eT>&& memory, arma::Mat<eT>&& output)
{
  // Pass the input through linear layer.
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, inputLinear))),
      inputLinear);

  arma::mat& lOutput = boost::apply_visitor(outputParameterVisitor,
    inputLinear);

  // Build kT with non linearity.
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(0, 0,
      memSize - 1, 0)), std::move(boost::apply_visitor(outputParameterVisitor,
      kTNonLinear))), kTNonLinear);
  const arma::mat& kT = boost::apply_visitor(outputParameterVisitor, kTNonLinear);

  output = arma::normalise(memory, 2, 1) * arma::normalise(kT);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* output */,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy,
  arma::Mat<eT>&& g,
  arma::Mat<eT>&& gM)
{
  arma::mat nMemory = memory;

  const arma::mat& kT = boost::apply_visitor(outputParameterVisitor, kTNonLinear);

  double kTNorm = arma::norm(kT);

  arma::mat nKt = kT / kTNorm;

  // Error of memory with normalization.
  gM = gy * arma::trans(nKt);

  // Error of memory without normalization.
  size_t rowIndex = 0;
  gM.each_row([&] (arma::rowvec& v)
  {
    double n = arma::norm(memory.row(rowIndex));
    nMemory.row(rowIndex) /= n;
    v = (v - (nMemory.row(rowIndex) * arma::as_scalar(arma::sum(nMemory.row(rowIndex) % v)))) / n;

    rowIndex++;
  });

  // Error of Kt with normalization
  arma::mat dKt = arma::trans(nMemory) * gy;

  // // Error of Kt without normalization.
  dKt = (dKt - (nKt * arma::as_scalar(arma::sum(nKt % dKt)))) / kTNorm;

  // Backward pass through Kt gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, kTNonLinear)), std::move(dKt),
      std::move(boost::apply_visitor(deltaVisitor, kTNonLinear))),
      kTNonLinear);

  prevError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor, kTNonLinear);

  // Backward pass through linear gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputLinear)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, inputLinear))),
      inputLinear);

  g = boost::apply_visitor(deltaVisitor, inputLinear);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(prevError)),
      inputLinear);
}

template<typename InputDataType, typename OutputDataType>
void MemoryHead<InputDataType, OutputDataType>::ResetCell()
{
  prevWeights.clear();
  prevWeights.push_back(arma::zeros<arma::mat>(outSize, 1));
  weightsBackwardIterator = prevWeights.end();

  prevError = arma::zeros<arma::mat>((memSize), 1);

  lWDash.clear();
  lGammaT.clear();
  lWTilde.clear();
  lShiftMatrix.clear();
  lWg.clear();
  lSt.clear();
  lGt.clear();
  lWe.clear();
  lWc.clear();
  lBt.clear();
  lConsineT.clear();

  bWdash = lWDash.end();
  bGammaT = lGammaT.end();
  bWTilde = lWTilde.end();
  bShiftMatrix = lShiftMatrix.end();
  bWg = lWg.end();
  bSt = lSt.end();
  bGt = lGt.end();
  bWe = lWe.end();
  bWc = lWc.end();
  bBt = lBt.end();
  bCosineT = lConsineT.end();
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MemoryHead<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{
}

} // namespace ann
} // namespace mlpack

#endif
