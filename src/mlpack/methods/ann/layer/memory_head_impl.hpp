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
  inputLinear = new Linear<>(inSize, (memSize) + (1) + (1) +
    (2 * shiftSize + 1) + (1));

  // kT non linearity.
  kTNonLinear = new TanHLayer<>();

  network.push_back(inputLinear);
  network.push_back(kTNonLinear);

  prevWeights.push_back(arma::zeros<arma::mat>(outSize, 1));
  weightsBackwardIterator = prevWeights.end();

  prevError = arma::zeros<arma::mat>((memSize) + (1) + (1) +
    (2 * shiftSize + 1) + (1), 1);

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
  arma::mat& kT = boost::apply_visitor(outputParameterVisitor, kTNonLinear);

  // Build bT with non linearity
  lBt.push_back(bTNonLinear.Fn(arma::as_scalar(lOutput.submat(memSize, 0,
    memSize, 0))));
  const double& bT = lBt.back();

  // Build gT with non linearity
  lGt.push_back(gTNonLinear.Fn(arma::as_scalar(lOutput.submat(memSize + 1, 0,
    memSize + 1, 0))));
  const double& gT = lGt.back();

  // Build sT with non linearity
  arma::vec temp = arma::exp(lOutput.submat(memSize + 2, 0,
    memSize + 2 + 2 * shiftSize, 0));
  temp = temp / arma::as_scalar(arma::sum(temp));
  lSt.push_back(std::move(temp));
  const arma::vec& sT = lSt.back();

  // Build gammaT with non linearity
  lGammaT.push_back(gammaTNonLinear.Fn(arma::as_scalar(
    lOutput.submat(memSize + 2 + 2 * shiftSize + 1, 0,
    memSize + 2 + 2 * shiftSize + 1, 0))));
  const double& gammaT = lGammaT.back();

  // Perform cosine similarity with memory content
  lConsineT.push_back(arma::normalise(memory, 1) * arma::normalise(kT));
  const arma::vec& cosSimilarity = lConsineT.back();

  // Build wC with bT and softmax
  lWe.push_back(arma::exp(bT * cosSimilarity));
  const arma::vec& wE = lWe.back();

  lWc.push_back(wE / arma::as_scalar(arma::sum(wE)));
  const arma::vec& wC = lWc.back();

  // Build wG with gT
  lWg.push_back(prevWeights.back() + arma::as_scalar(gT) *
    (wC - prevWeights.back()));
  const arma::vec& wG = lWg.back();

  // Perform circular convolution with sT
  arma::mat shiftVec = arma::shift(arma::flipud(sT), 1, 1);
  size_t numRep = memSize / (2 * shiftSize + 1);
  if(numRep > 1)
  {
    shiftVec = arma::repmat(shiftVec, numRep, 1);
  }

  lShiftMatrix.push_back(arma::mat(outSize, outSize, arma::fill::none));
  arma::mat& shiftMatrix = lShiftMatrix.back();

  shiftMatrix.each_col([&](arma::vec& a)
  {
    a = shiftVec.submat(0, 0, wG.n_rows - 1, 0);
    shiftVec = arma::shift(std::move(shiftVec), 1, 1);
  });

  lWTilde.push_back(arma::trans(arma::trans(wG) * shiftMatrix));
  const arma::vec& wTilde = lWTilde.back();

  // Sharpening
  lWDash.push_back(arma::pow(wTilde, gammaT + 1));
  const arma::vec& wDash = lWDash.back();
  output = wDash / arma::as_scalar(arma::sum(wDash));

  if (!deterministic)
  {
    prevWeights.push_back(output);
  }
  else
  {
    prevWeights.back() = output;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* input */,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy, arma::Mat<eT>&& g, arma::Mat<eT>&& gM)
{
  double sum = 0;
  double sum2 = 0;

  arma::vec dWDash(outSize, 1, arma::fill::none);

  if(weightsBackwardIterator == prevWeights.end())
  {
    bWdash = (--lWDash.end());
    bGammaT = (--lGammaT.end());
    bWTilde = (--lWTilde.end());
    bShiftMatrix = (--lShiftMatrix.end());
    bWg = (--lWg.end());
    bSt = (--lSt.end());
    bGt = (--lGt.end());
    bWc = (--lWc.end());
    bWe = (--lWe.end());
    bBt = (--lBt.end());
    bCosineT = (--lConsineT.end());

    weightsBackwardIterator = --(--prevWeights.end());

    // W_t = (wDash_t / sum(wDash_t)).
    sum = arma::as_scalar(arma::sum(*bWdash));

    auto gyIt = gy.begin();

    double sum2 = arma::as_scalar(arma::sum(*bWdash % gy)) / (sum * sum);
    dWDash.for_each([&] (double& val)
    {
      val = (*gyIt / sum) - sum2;
      gyIt++;
    });
  }
  else
  {
    // W_t = (wDash_t / sum(wDash_t)).
    double sum = arma::as_scalar(arma::sum(*bWdash));

    auto gyIt = gy.begin();
    auto predDWIt = prevDW.begin();

    sum2 = arma::as_scalar(arma::sum(*bWdash % gy)) / (sum * sum);
    dWDash.for_each([&] (double& val)
    {
      val = ((*gyIt + *predDWIt) / sum) - sum2;
      gyIt++;
      predDWIt++;
    });
  }

  // Load parameters of this pass.
  const arma::vec& wDash = *bWdash;
  const double& gammaT = *bGammaT;
  const double& gT = *bGt;
  const arma::vec& wTilde = *bWTilde;
  const arma::mat& shiftMatrix = *bShiftMatrix;
  const arma::vec& wG = *bWg;
  const arma::vec& sT = *bSt;
  const arma::vec& wC = *bWc;
  const arma::vec& wE = *bWe;
  const double& bT = *bBt;
  const arma::vec& consineT = *bCosineT;

  arma::vec dWTilde = (gammaT + 1) * (dWDash % arma::pow(wTilde, gammaT));

  // delta of gammaT
  prevError(outSize + 2 + 2 * shiftSize + 1, 0) = gammaTNonLinear.Deriv(
    boost::apply_visitor(outputParameterVisitor, inputLinear)
    (memSize + 2 + 2 * shiftSize + 1, 0)) *
    arma::as_scalar(arma::sum(dWDash % wDash % arma::log(wTilde)));

  arma::vec dWg = shiftMatrix * dWTilde;

  arma::mat dShiftMatrix = wG * arma::trans(dWTilde);

  size_t rowIndex = 2 * shiftSize + 1;

  while(rowIndex < shiftMatrix.n_rows)
  {
    const arma::mat& toAdd = dShiftMatrix.submat(rowIndex, 0,
      std::min(rowIndex + 2 * shiftSize, (size_t)shiftMatrix.n_rows - 1),
      shiftMatrix.n_cols - 1);

    dShiftMatrix.submat(0, 0, toAdd.n_rows - 1, shiftMatrix.n_cols - 1) +=
      toAdd;

    rowIndex += 2 * shiftSize + 1;
  }

  size_t colIndex = 2 * shiftSize + 1;

  while(colIndex < shiftMatrix.n_cols)
  {
    const arma::mat& toAdd = dShiftMatrix.submat(0, colIndex, 2 * shiftSize,
      std::min(colIndex + 2 * shiftSize, (size_t)shiftMatrix.n_cols - 1));
    dShiftMatrix.submat(0, 0, 2 * shiftSize, toAdd.n_cols - 1) += toAdd;
  }

  arma::mat sDShiftMatrix = dShiftMatrix.submat(0, 0, 2 * shiftSize,
    2 * shiftSize);

  arma::vec dSt = arma::zeros(2 * shiftSize + 1);

  sDShiftMatrix.each_col([&](arma::vec& v)
  {
    dSt = std::move(arma::shift(std::move(dSt), 1));
    dSt += v;
  });

  // dSt
  dSt = sT % arma::flipud(std::move(dSt));
  prevError.submat(memSize + 2, 0, memSize + 2 + 2 * shiftSize, 0) =
    dSt - arma::sum(dSt * arma::trans(sT), 1);

  arma::vec dWc = gT * dWg;

  prevDW = (1 - gT) * dWg;

  //d_gt
  prevError(memSize + 1, 0) = gTNonLinear.Deriv(
    boost::apply_visitor(outputParameterVisitor, inputLinear)(memSize + 1, 0)) *
    arma::as_scalar(arma::sum(dWg % (wC - *weightsBackwardIterator)));

  sum = arma::as_scalar(arma::sum(wE));
  arma::vec dWe = dWc / sum;
  sum = arma::as_scalar(arma::sum(wE % dWc)) / (sum * sum);
  dWe.for_each([&] (double& val)
  {
    val -= sum;
  });

  arma::vec dCosineT = (dWe % wE) * bT;

  // d_bt
  prevError(memSize, 0) = bTNonLinear.Deriv(
    boost::apply_visitor(outputParameterVisitor, inputLinear)(memSize, 0)) *
    arma::as_scalar(arma::sum(dWe % consineT % wE));

  // Differentiation of kT with normalisation.
  arma::vec dKt = arma::trans(arma::normalise(memory, 1)) * dCosineT;

  const arma::vec& kT = boost::apply_visitor(outputParameterVisitor,
    kTNonLinear);

  double kTNorm = arma::norm(kT);

  // Differentiation without normalization.
  dKt = arma::sum((arma::eye(memSize, memSize) - (kT * arma::trans(kT) /
    (kTNorm * kTNorm))) / kTNorm, 1) % dKt;

  // Differentiation of Memory with normalization.
  gM = dCosineT * arma::trans(arma::normalise(kT));

  // Differentiation of Memory without normalization.
  size_t memRow = 0;
  gM.each_row([&] (arma::rowvec& v)
  {
    double n = arma::norm(memory.row(memRow));
    v = arma::sum((arma::eye(memSize, memSize) -
        (arma::trans(memory.row(memRow)) * memory.row(memRow) /
        (n * n))) / n) % v;

    memRow++;
  });

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, kTNonLinear)), std::move(dKt),
      std::move(boost::apply_visitor(deltaVisitor, kTNonLinear))),
      kTNonLinear);

  prevError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor,
    kTNonLinear);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputLinear)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, inputLinear))),
      inputLinear);

  g = boost::apply_visitor(deltaVisitor, inputLinear);

  bWdash--;
  bGammaT--;
  bWTilde--;
  bShiftMatrix--;
  bWg--;
  bSt--;
  bGt--;
  bWc--;
  bWe--;
  bBt--;
  bCosineT--;

  weightsBackwardIterator--;
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

  prevError = arma::zeros<arma::mat>((memSize) + (1) + (1) +
    (2 * shiftSize + 1) + (1), 1);

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
