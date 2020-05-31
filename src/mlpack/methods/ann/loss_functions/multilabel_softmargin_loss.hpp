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
namespace ann /** Artificial Neural Network. */ {

/**
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MultiLabelSoftMarginLoss
{
 public:
  /**
   * Create the MultiLabelSoftMarginLoss object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If
   *                  true, 'sum' reduction is used and the output will be
   *                  summed. It is set to true by default.
   * @param weights A manual rescaling weight given to each class. It is a
   *                (1, numClasses) row vector.
   */
  MultiLabelSoftMarginLoss(const bool reduction = true,
                           const arma::rowvec& weights = arma::rowvec());

  /**
   * Computes the Multi Label Soft Margin Loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector with same shape as input.
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

  //! Get the weights assigned to each class.
  const arma::rowvec& ClassWeights() const { return classWeights; }
  //! Modify the weights assigned to each class.
  arma::rowvec& ClassWeights() { return classWeights; }

  //! Get the type of reduction used.
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The boolean value that tells if reduction is sum or mean.
  bool reduction;

  //! A (1, numClasses) shaped vector with weights for each class.
  arma::rowvec classWeights;

  // An internal parameter used during initialisation of class weights.
  bool weighted;
}; // class MultiLabelSoftMarginLoss

} // namespace ann
} // namespace mlpack

// include implementation.
#include "multilabel_softmargin_loss_impl.hpp"

#endif
