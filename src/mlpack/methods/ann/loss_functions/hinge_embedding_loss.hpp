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
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
        typename InputDataType = arma::mat,
        typename OutputDataType = arma::mat
>
class HingeEmbeddingLoss
{
 public:
  /**
   * Create the Hinge Embedding object.
   */
  HingeEmbeddingLoss();

  /**
   * Computes the Hinge Embedding loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target Target data to compare with.
   */
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType& input,
                                        const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  /**
   * Serialize the loss function.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class HingeEmbeddingLoss

} // namespace ann
} // namespace mlpack

// include implementation
#include "hinge_embedding_loss_impl.hpp"

#endif
