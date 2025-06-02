/**
 * @file methods/ann/loss_functions/multilabel_softmargin_loss.hpp
 * @author Anjishnu Mukherjee
 *
 * Definition of the Multi Label Soft Margin Loss function.
 *
 * It is a criterion that optimizes a multi-label one-versus-all loss based on
 * max-entropy, between input x and target y of size (N, C) where N is the
 * batch size and C is the number of classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ANN_LOSS_FUNCTION_MULTILABEL_SOFTMARGIN_LOSS_HPP
#define MLPACK_ANN_LOSS_FUNCTION_MULTILABEL_SOFTMARGIN_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Multi-label Soft Margin Loss function.
 *
 * It is a criterion that optimizes a multi-label one-versus-all loss based on
 * max-entropy, between input x and target y of size (N, C) where N is the
 * batch size and C is the number of classes.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class MultiLabelSoftMarginLossType
{
 public:
  /**
   * Create the MultiLabelSoftMarginLossType object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If
   *                  true, 'sum' reduction is used and the output will be
   *                  summed. It is set to true by default.
   * @param weights A manual rescaling weight given to each class. It is a
   *                (1, numClasses) row vector.
   */
  MultiLabelSoftMarginLossType(
      const bool reduction = true,
      const arma::Row<typename MatType::elem_type>& weights =
          arma::Row<typename MatType::elem_type>());

  /**
   * Computes the Multi Label Soft Margin Loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector with same shape as input.
   */
  typename MatType::elem_type Forward(const MatType& input,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  void Backward(const MatType& input,
                const MatType& target,
                MatType& output);

  //! Get the weights assigned to each class.
  const arma::Row<typename MatType::elem_type>& ClassWeights() const
  {
    return classWeights;
  }
  //! Modify the weights assigned to each class.
  arma::Row<typename MatType::elem_type>& ClassWeights()
  {
    return classWeights;
  }

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
  //! The boolean value that tells if reduction is sum or mean.
  bool reduction;

  //! A (1, numClasses) shaped vector with weights for each class.
  arma::Row<typename MatType::elem_type> classWeights;

  // An internal parameter used during initialisation of class weights.
  bool weighted;
}; // class MultiLabelSoftMarginLossType

// Default typedef for typical `arma::mat` usage.
using MultiLabelSoftMarginLoss = MultiLabelSoftMarginLossType<arma::mat>;

} // namespace mlpack

// include implementation.
#include "multilabel_softmargin_loss_impl.hpp"

#endif
