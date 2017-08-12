/**
 * @file ham_unit.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the HAMUnit class, which implements a Hierarchical Attentive
 * Memory unit as described in https://arxiv.org/abs/1602.03218.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_AUGMENTED_HAM_UNIT_HPP
#define MLPACK_METHODS_ANN_AUGMENTED_HAM_UNIT_HPP

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

#include "tree_memory.hpp"

using namespace mlpack::ann;
using namespace mlpack::optimization;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {

template<
  typename E = FFN<MeanSquaredError<>>,
  typename J = FFN<MeanSquaredError<>>,
  typename S = FFN<MeanSquaredError<>>,
  typename W = FFN<MeanSquaredError<>>
>
class HAMUnit
{
 public:
  HAMUnit(size_t memorySize,
          size_t memoryDim,
          E& embed,
          J& join,
          S& search,
          W& write);

  void Evaluate(const arma::mat& predictors,
                const arma::mat& responses);

  TreeMemory<double, J, W> Memory() const { return memory; }
  TreeMemory<double, J, W>& Memory() { return memory; }

  void Predict(arma::mat&& input, arma::mat&& output);
 private:
  arma::vec Attention();

  void Forward(arma::mat&& input, arma::mat&& output);

  void Backward(arma::mat&& input,
                arma::mat&& gy,
                arma::mat&& g);

  void Gradient(arma::mat&& input,
                arma::mat&& error,
                arma::mat&& gradient);

  TreeMemory<double, J, W> memory;
  size_t memorySize, memoryDim;
  S search;
  E embed;

  // Currently processed sequence.
  arma::mat sequence;
  size_t t;
};
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "ham_unit_impl.hpp"
#endif
