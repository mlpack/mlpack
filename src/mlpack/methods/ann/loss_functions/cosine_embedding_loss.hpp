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
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  CosineEmbeddingLossType(const double margin = 0.0,
                      const bool similarity = true,
                      const bool reduction = true);

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

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

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

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class CosineEmbeddingLossType

// Default typedef for typical `arma::mat` usage.
using CosineEmbeddingLoss = CosineEmbeddingLossType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "cosine_embedding_loss_impl.hpp"

#endif
