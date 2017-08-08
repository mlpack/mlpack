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

#include "tree_memory.hpp"

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {

class HAMUnit {
 public:
  HAMUnit(size_t memorySize,
          LayerTypes& embed,
          LayerTypes& join,
          LayerTypes& search,
          LayerTypes& write);
  
  template<
    template<typename, typename...> class OptimizerType =
        mlpack::optimization::StandardSGD,
    typename... OptimizerTypeArgs
  >
  void Train(const MatType& predictors,
             const MatType& responses,
             OptimizerType<Controller> optimizer,
             double gamma);

  void Evaluate(const arma::mat& predictors,
                const arma::mat& responses);

  TreeMemory<double, LayerTypes, LayerTypes> Memory() const { return memory; }
  TreeMemory<double, LayerTypes, LayerTypes>& Memory() { return memory; }
 private:
  arma::mat Attention() const;

  void Forward(arma::mat&& input, arma::mat&& output);

  void Backward(arma::mat&& input,
                arma::mat&& gy,
                arma::mat&& g);

  void Gradient(arma::mat&& input,
                arma::mat&& error,
                arma::mat&& gradient);

  TreeMemory<double, LayerTypes, LayerTypes> memory;
  size_t memorySize;
  LayerTypes search;
  LayerTypes embed;

  // Currently prcessed sequence.
  arma::mat sequence;
  size_t t;
};
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "ham_unit_impl.hpp"
#endif
