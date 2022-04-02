/**
 * @file methods/ann/loss_functions/hinge_embedding_loss.hpp
 * @author Lakshya Ojha
 *
 * Definition of the Hinge Embedding Loss Function.
 * The Hinge Embedding loss function is often used to improve performance
 * in semi-supervised learning or to learn nonlinear embeddings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_EMBEDDING_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_EMBEDDING_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Hinge Embedding loss function is often used to compute the loss
 * between y_true and y_pred.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class HingeEmbeddingLossType
{
 public:
  /**
   * Create the Hinge Embedding object.
   */
  HingeEmbeddingLossType();

  /**
   * Computes the Hinge Embedding loss function.
   *
   * @param prediction Prediction used for evaluating the specified loss
   *     function.
   * @param target Target data to compare with.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Prediction used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  /**
   * Serialize the loss function.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) { }
}; // class HingeEmbeddingLossType

// Default typedef for typical `arma::mat` usage.
typedef HingeEmbeddingLossType<arma::mat> HingeEmbeddingLoss;

} // namespace ann
} // namespace mlpack

// include implementation
#include "hinge_embedding_loss_impl.hpp"

#endif
