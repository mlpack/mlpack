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
  // Build linear for K_t + B_t + G_t + S_t + Gamma_t
  inputLinear = new Linear<>(inSize, (memSize) + (1) + (1) +
    (2 * shiftSize + 1) + (1));

  // K_t non linearity.
  kTNonLinear = new TanHLayer<>();

  network.push_back(inputLinear);
  network.push_back(kTNonLinear);

  prevWeights.push_back(arma::zeros<arma::mat>(outSize, 1));
  weightsBackwardIterator = prevWeights.end();

  prevError = arma::zeros<arma::mat>((memSize) + (1) + (1) +
    (2 * shiftSize + 1) + (1), 1);

  bWDash = l_w_dash.end();
  b_gamma_t = l_gamma_t.end();
  b_w_tilde = l_w_tilde.end();
  b_shiftMatrix = l_shiftMatrix.end();
  b_w_g = l_w_g.end();
  b_s_t = l_s_t.end();
  b_g_t = l_g_t.end();
  b_w_e = l_w_e.end();
  b_w_c = l_w_c.end();
  b_b_t = l_b_t.end();
  b_cosine_t = l_cosine_t.end();
  b_memory_t = l_memory_t.end();
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::mat&& memory, arma::Mat<eT>&& output)
{
  // Pass the input through linear layer.
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, inputLinear))),
      inputLinear);

  arma::mat& lOutput = boost::apply_visitor(outputParameterVisitor, inputLinear);

  // Build K_t with non linearity.
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(0, 0, memSize - 1, 0)),
      std::move(boost::apply_visitor(outputParameterVisitor, kTNonLinear))),
      kTNonLinear);
  arma::mat& k_t = boost::apply_visitor(outputParameterVisitor, kTNonLinear);

  // Build B_t with non linearity
  l_b_t.push_back(b_t_non_linear.Fn(arma::as_scalar(lOutput.submat(memSize, 0, memSize, 0))));
  const double& b_t = l_b_t.back();

  // Build G_t with non linearity
  l_g_t.push_back(g_t_non_linear.Fn(arma::as_scalar(lOutput.submat(memSize + 1, 0, memSize + 1, 0))));
  const double& g_t = l_g_t.back();

  // Build S_t with non linearity
  arma::vec temp = arma::exp(lOutput.submat(memSize + 2, 0, memSize + 2 + 2 * shiftSize, 0));
  temp = temp / arma::as_scalar(arma::sum(temp));
  l_s_t.push_back(std::move(temp));
  const arma::vec& s_t = l_s_t.back();

  // Build gamma_t with non linearity
  l_gamma_t.push_back(gamma_t_non_linear.Fn(arma::as_scalar(lOutput.submat(memSize + 2 + 2 * shiftSize + 1, 0, memSize + 2 + 2 * shiftSize + 1, 0))));
  const double& gamma_t = l_gamma_t.back();

  // Perform cosine similarity with memory content
  l_memory_t.push_back(arma::normalise(memory));
  l_cosine_t.push_back(l_memory_t.back() * arma::normalise(k_t));// / arma::norm(k_t));
  const arma::vec& cosSimilarity = l_cosine_t.back();

  // Build w_c with b_t and softmax
  l_w_e.push_back(arma::exp(b_t * cosSimilarity));
  const arma::vec& w_e = l_w_e.back();

  l_w_c.push_back(w_e / arma::as_scalar(arma::sum(w_e)));
  const arma::vec& w_c = l_w_c.back();

  // Build w_g with g_t
  l_w_g.push_back(prevWeights.back() + arma::as_scalar(g_t) * (w_c - prevWeights.back()));
  const arma::vec& w_g = l_w_g.back();

  // Perform circular convolution with s_t
  arma::mat shiftVec = arma::shift(arma::flipud(s_t), 1, 1);
  size_t numRep = memSize / (2 * shiftSize + 1);
  if(numRep > 1)
  {
    shiftVec = arma::repmat(shiftVec, numRep, 1);
  }

  l_shiftMatrix.push_back(arma::mat(outSize, outSize, arma::fill::none));
  arma::mat& shiftMatrix = l_shiftMatrix.back();

  shiftMatrix.each_col([&](arma::vec& a)
  {
    a = shiftVec.submat(0, 0, w_g.n_rows - 1, 0);
    shiftVec = arma::shift(std::move(shiftVec), 1, 1);
  });

  l_w_tilde.push_back(arma::trans(arma::trans(w_g) * shiftMatrix));
  const arma::vec& w_tilde = l_w_tilde.back();

  // Sharpening
  l_w_dash.push_back(arma::pow(w_tilde, gamma_t + 1));
  const arma::vec& w_dash = l_w_dash.back();
  output = w_dash / arma::as_scalar(arma::sum(w_dash));

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
void MemoryHead<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  double sum = 0;
  double sum2 = 0;

  arma::vec d_w_dash(outSize, 1, arma::fill::none);

  if(b_memory_t == l_memory_t.end())
  {
    bWDash = (--l_w_dash.end());
    b_gamma_t = (--l_gamma_t.end());
    b_w_tilde = (--l_w_tilde.end());
    b_shiftMatrix = (--l_shiftMatrix.end());
    b_w_g = (--l_w_g.end());
    b_s_t = (--l_s_t.end());
    b_g_t = (--l_g_t.end());
    b_w_c = (--l_w_c.end());
    b_w_e = (--l_w_e.end());
    b_b_t = (--l_b_t.end());
    b_cosine_t = (--l_cosine_t.end());
    b_memory_t = (--l_memory_t.end());

    weightsBackwardIterator = --(--prevWeights.end());

    // W_t = (W_dash_t / sum(W_dash_t)).
    sum = arma::as_scalar(arma::sum(*bWDash));

    auto gyIt = gy.begin();

    double sum2 = arma::as_scalar(arma::sum(*bWDash % gy)) / (sum * sum);
    d_w_dash.for_each([&] (double& val)
    {
      val = (*gyIt / sum) - sum2;
      gyIt++;
    });
  }
  else
  {
    // W_t = (W_dash_t / sum(W_dash_t)).
    double sum = arma::as_scalar(arma::sum(*bWDash));

    auto gyIt = gy.begin();
    auto predDWIt = prev_d_w.begin();

    sum2 = arma::as_scalar(arma::sum(*bWDash % gy)) / (sum * sum);
    d_w_dash.for_each([&] (double& val)
    {
      val = ((*gyIt + *predDWIt) / sum) - sum2;
      gyIt++;
      predDWIt++;
    });
  }

  // Load parameters of this pass.
  const arma::vec& w_dash = *bWDash;
  const double& gamma_t = *b_gamma_t;
  const double& g_t = *b_g_t;
  const arma::vec& w_tilde = *b_w_tilde;
  const arma::mat& shiftMatrix = *b_shiftMatrix;
  const arma::vec& w_g = *b_w_g;
  const arma::vec& s_t = *b_s_t;
  const arma::vec& w_c = *b_w_c;
  const arma::vec& w_e = *b_w_e;
  const double& b_t = *b_b_t;
  const arma::vec& cosine_t = *b_cosine_t;
  const arma::mat& memory_t = *b_memory_t;

  arma::vec d_w_tilde = (gamma_t + 1) * (d_w_dash % arma::pow(w_tilde, gamma_t));

  // delta of gamma_t
  prevError(outSize + 2 + 2 * shiftSize + 1, 0) = gamma_t_non_linear.Deriv(boost::apply_visitor(outputParameterVisitor, inputLinear)(memSize + 2 + 2 * shiftSize + 1, 0)) * arma::as_scalar(arma::sum(d_w_dash % w_dash % arma::log(w_tilde)));

  arma::vec d_w_g = shiftMatrix * d_w_tilde;

  arma::mat d_shift_matrix = w_g * arma::trans(d_w_tilde);

  size_t rowIndex = 2 * shiftSize + 1;

  while(rowIndex < shiftMatrix.n_rows)
  {
    const arma::mat& toAdd = d_shift_matrix.submat(rowIndex, 0, std::min(rowIndex + 2 * shiftSize, (size_t)shiftMatrix.n_rows - 1), shiftMatrix.n_cols - 1);
    d_shift_matrix.submat(0, 0, toAdd.n_rows - 1, shiftMatrix.n_cols - 1) += toAdd;
    rowIndex += 2 * shiftSize + 1;
  }

  size_t colIndex = 2 * shiftSize + 1;

  while(colIndex < shiftMatrix.n_cols)
  {
    const arma::mat& toAdd = d_shift_matrix.submat(0, colIndex, 2 * shiftSize, std::min(colIndex + 2 * shiftSize, (size_t)shiftMatrix.n_cols - 1));
    d_shift_matrix.submat(0, 0, 2 * shiftSize, toAdd.n_cols - 1) += toAdd;
  }

  arma::mat s_d_shift_matrix = d_shift_matrix.submat(0, 0, 2 * shiftSize, 2 * shiftSize);

  arma::vec d_st = arma::zeros(2 * shiftSize + 1);

  s_d_shift_matrix.each_col([&](arma::vec& v)
  {
    d_st = std::move(arma::shift(std::move(d_st), 1));
    d_st += v;
  });

  // d_st
  d_st = s_t % arma::flipud(std::move(d_st));
  prevError.submat(memSize + 2, 0, memSize + 2 + 2 * shiftSize, 0) = d_st - arma::sum(d_st * arma::trans(s_t), 1);

  arma::vec d_w_c = g_t * d_w_g;

  prev_d_w = (1 - g_t) * d_w_g;

  //d_gt
  prevError(memSize + 1, 0) = g_t_non_linear.Deriv(boost::apply_visitor(outputParameterVisitor, inputLinear)(memSize + 1, 0)) * arma::as_scalar(arma::sum(d_w_g % (w_c - *weightsBackwardIterator)));

  sum = arma::as_scalar(arma::sum(w_e));
  arma::vec d_w_e = d_w_c / sum;
  sum = arma::as_scalar(arma::sum(w_e % d_w_c)) / (sum * sum);
  d_w_e.for_each([&] (double& val)
  {
    val -= sum;
  });

  arma::vec d_cosine_t = (d_w_e % w_e) * b_t;

  // d_bt
  prevError(memSize, 0) = b_t_non_linear.Deriv(boost::apply_visitor(outputParameterVisitor, inputLinear)(memSize, 0)) * arma::as_scalar(arma::sum(d_w_e % cosine_t % w_e));

  // Derivative with normalisation.
  arma::vec d_k_t = arma::trans(memory_t) * d_cosine_t;

  const arma::vec& k_t = boost::apply_visitor(outputParameterVisitor, kTNonLinear);

  double k_t_norm = arma::norm(k_t);

  // Derivative without normalization.
  d_k_t = arma::sum((arma::eye(memSize, memSize) - (k_t * arma::trans(k_t) / (k_t_norm * k_t_norm))) / k_t_norm, 1) % d_k_t;

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, kTNonLinear)), std::move(d_k_t),
      std::move(boost::apply_visitor(deltaVisitor, kTNonLinear))),
      kTNonLinear);

  prevError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor, kTNonLinear);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputLinear)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, inputLinear))),
      inputLinear);

  g = boost::apply_visitor(deltaVisitor, inputLinear);

  bWDash--;
  b_gamma_t--;
  b_w_tilde--;
  b_shiftMatrix--;
  b_w_g--;
  b_s_t--;
  b_g_t--;
  b_w_c--;
  b_w_e--;
  b_b_t--;
  b_cosine_t--;
  b_memory_t--;

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

  l_w_dash.clear();
  l_gamma_t.clear();
  l_w_tilde.clear();
  l_shiftMatrix.clear();
  l_w_g.clear();
  l_s_t.clear();
  l_g_t.clear();
  l_w_e.clear();
  l_w_c.clear();
  l_b_t.clear();
  l_cosine_t.clear();
  l_memory_t.clear();

  bWDash = l_w_dash.end();
  b_gamma_t = l_gamma_t.end();
  b_w_tilde = l_w_tilde.end();
  b_shiftMatrix = l_shiftMatrix.end();
  b_w_g = l_w_g.end();
  b_s_t = l_s_t.end();
  b_g_t = l_g_t.end();
  b_w_e = l_w_e.end();
  b_w_c = l_w_c.end();
  b_b_t = l_b_t.end();
  b_cosine_t = l_cosine_t.end();
  b_memory_t = l_memory_t.end();
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
