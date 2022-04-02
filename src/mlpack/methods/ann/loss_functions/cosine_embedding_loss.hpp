/**
 * @file methods/ann/loss_functions/cosine_embedding_loss.hpp
 * @author Kartik Dutt
 *
 * Definition of the Cosine Embedding loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Cosine Embedding Loss function is used for measuring whether two inputs are
 * similar or dissimilar, using the cosine distance, and is typically used
 * for learning nonlinear embeddings or semi-supervised learning.
 *
 * @f{eqnarray*}{
 * f(x) = 1 - cos(x1, x2) , for y = 1
 * f(x) = max(0, cos(x1, x2) - margin) , for y = -1
 * @f}
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class CosineEmbeddingLossType
{
 public:
  /**
   * Create the CosineEmbeddingLossType object.
   *
   * @param margin Increases cosine distance in case of dissimilarity.
   *               Refer definition of cosine-embedding-loss above.
   * @param similarity Determines whether to use similarity or dissimilarity for
   *                   comparision.
   * @param takeMean Boolean variable to specify whether to take mean or not.
   *                 Specifies reduction method i.e. sum or mean corresponding
   *                 to 0 and 1 respectively. Default value = 0.
   */
  CosineEmbeddingLossType(const double margin = 0.0,
                      const bool similarity = true,
                      const bool takeMean = false);

  /**
   * Ordinary feed forward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Get the value of takeMean.
  bool TakeMean() const { return takeMean; }
  //! Modify the value of takeMean.
  bool& TakeMean() { return takeMean; }

  //! Get the value of margin.
  double Margin() const { return margin; }
  //! Modify the value of takeMean.
  double& Margin() { return margin; }

  //! Get the value of similarity hyperparameter.
  bool Similarity() const { return similarity; }
  //! Modify the value of takeMean.
  bool& Similarity() { return similarity; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored value of margin hyper-parameter.
  double margin;

  //! Locally-stored value of similarity hyper-parameter.
  bool similarity;

  //! Locally-stored value of takeMean hyper-parameter.
  bool takeMean;
}; // class CosineEmbeddingLossType

typedef CosineEmbeddingLossType<arma::mat> CosineEmbeddingLoss;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "cosine_embedding_loss_impl.hpp"

#endif
