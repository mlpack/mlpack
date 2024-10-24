/**
 * @file methods/ann/layer/reinforce_normal.hpp
 * @author Marcus Edel
 *
 * Definition of the ReinforceNormalLayer class, which implements the REINFORCE
 * algorithm for the normal distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_HPP
#define MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the reinforce normal layer. The reinforce normal layer
 * implements the REINFORCE algorithm for the normal distribution.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class ReinforceNormalType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the ReinforceNormal object.
   *
   * @param stdev Standard deviation used during the forward and backward pass.
   */
  ReinforceNormalType(const double stdev = 1.0);

  //! Clone the ReinforceNormalType object. This handles polymorphism correctly.
  ReinforceNormalType* Clone() const { return new ReinforceNormalType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param * (gy) The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& /* gy */,
                OutputType& g);

  //! Get the value of the reward parameter.
  double const& Reward() const { return reward; }
  //! Modify the value of the deterministic parameter.
  double& Reward() { return reward; }

  //! Get the standard deviation used during forward and backward pass.
  double const& StandardDeviation() const { return stdev; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Standard deviation used during the forward and backward pass.
  double stdev;

  //! Locally-stored reward parameter.
  double reward;

  //!  Locally-stored output module parameter parameters.
  std::vector<InputType> moduleInputParameter;

  //! If true use maximum a posteriori during the forward pass.
  bool deterministic;
}; // class ReinforceNormalType.

// Standard ReinforceNormal layer.
using ReinforceNormal = ReinforceNormalType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "reinforce_normal_impl.hpp"

#endif
