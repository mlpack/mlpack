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
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include "tree_memory.hpp"

using namespace mlpack::ann;
using namespace mlpack::optimization;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {

/**
 * The class that implements the HAM unit.
 */
template<
  typename E = FFN<MeanSquaredError<>>,
  typename J = FFN<MeanSquaredError<>>,
  typename S = FFN<MeanSquaredError<>>,
  typename W = FFN<MeanSquaredError<>>,
  typename C = FFN<CrossEntropyError<>>
>
class HAMUnit
{
 public:
  /**
   * Initialize the HAM unit components.
   * 
   * @param memorySize Number of leaf nodes used.
   * @param memoryDim Node value dimensionality.
   * @param embed Embed function which takes the raw data vector
   *              and creates the embedding for the memory.
   * @param join Join function which takes two memory vectors
   *             and evaluates the memory vector of their parent cell.
   * @param search Search function which evaluates the probability of turning
   *               left during the tree traversal.
   * @param write Write function which outputs the new cell value
   *              given old cell value and update value.
   * @param controller Controller function that emits the new sequence symbol
   *                   given memory vector obtained during memory tree traverse.
   */
  HAMUnit(size_t memorySize,
          size_t memoryDim,
          E& embed,
          J& join,
          S& search,
          W& write,
          C& controller);

  /**
   * Predict the responses to a given set of predictors.
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   */
  void Evaluate(const arma::mat& predictors,
                const arma::mat& responses);

  /**
   * Return the TreeMemory instance used in the unit.
   */
  TreeMemory<double, J, W> Memory() const { return memory; }
  /**
   * Modify the TreeMemory instance used in the unit.
   */
  TreeMemory<double, J, W>& Memory() { return memory; }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(arma::mat&& input, arma::mat&& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(arma::mat&& input,
                arma::mat&& gy,
                arma::mat&& g);

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameters; }

  /**
   * Reset the module information (weights/parameters).
   */
  void ResetParameters();
 private:
  /**
   * Compute the attention (probability distribution) over leaf nodes.
   * 
   * @param attention The output probability vector.
   */
  void Attention(arma::vec& attention);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(arma::mat&& input,
                arma::mat&& error,
                arma::mat&& gradient);

  //! Parameter matrix.
  arma::mat parameters;

  //! TreeMemory instance used in the unit.
  TreeMemory<double, J, W> memory;
  //! Number of used leaf nodes.
  size_t memorySize;
  //! Dimensionality of the single node value.
  size_t memoryDim;
  //! Search function.
  S search;
  //! Embed function.
  E embed;
  //! Controller function.
  C controller;

  //! Indicator if we already trained the model.
  bool reset;

  //! Currently processed sequence.
  arma::mat sequence;
  //! The index of currently processed element.
  size_t t;
};
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "ham_unit_impl.hpp"
#endif
