/**
 * @file multilabel_softmargin_loss.hpp
 * @author Anjishnu Mukherjee
 *
 * Definition of the Multi Label Soft Margin Loss function.
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
	 * @param weight A manual rescaling weight given to each class. Initialized to
	 *        1 by default.
	 * @param reduction Specifies the reduction to apply to the output. When true,
	 * 				'mean' reduction is used, where sum of the output will be divided by
	 * 				the number of elements in the output. When false, 'sum' reduction is
	 * 				used and the output will be summed.
   */

  MultiLabelSoftMarginLoss(const double weight = 1.0,
                           const bool reduction = true);

  /**
   * Computes the Multi Label Soft Margin Loss function.
   * This criterion optimizes a multi-label one-versus-all loss based
	 * on max-entropy, between input x and target y.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector with same shape as input.
   */

  template<typename InputType, typename TargetType>
  double Forward(const InputType& input, const TargetType& target);
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

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the weight.
  double Weight() const { return weight; }
  //! Modify the weight.
  double& Weight() { return weight; }

  //! Get the reduction.
  bool Reduction() const { return reduction; }
  //! Modify the reduction.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! The manual rescaling factor given to the loss.
  double weight;

  //! The weight for positive examples.
  double posWeight;

  //! The boolean value that tells if reduction is mean or sum.
  bool reduction;
}; // class MarginRankingLoss

} // namespace ann
} // namespace mlpack

// include implementation.
#include "multilabel_softmargin_loss_impl.hpp"

#endif
