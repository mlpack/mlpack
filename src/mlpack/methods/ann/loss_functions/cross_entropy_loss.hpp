/**
 * @file methods/ann/loss_functions/cross_entropy.hpp
 * @author Antoine Dubois
 *
 * Definition of the cross-entropy performance function between
 * p and q. 
 * !The target q must be passed as log-q to the Forward
 * and backward methods.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The cross-entropy performance function measures the Cross
 * Entropy between the target and the log-output.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class CrossEntropyLossType
{
 public:
  /**
   * Create the CrossEntropyLoss object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  CrossEntropyLossType(const bool reduction = true);

  /**
   * Computes the cross-entropy function. sum (p log(q))
   *
   * @param prediction Predictions of the log-target: log-q
   * @param target The target probability distribution: p
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction log(q), redictions of log(p).
   * @param target p the target distribution.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class CrossEntropyLossType

// Default typedef for typical `arma::mat` usage.
typedef CrossEntropyLossType<arma::mat> CrossEntropyLoss;

} // namespace mlpack

// Include implementation.
#include "cross_entropy_loss_impl.hpp"

#endif