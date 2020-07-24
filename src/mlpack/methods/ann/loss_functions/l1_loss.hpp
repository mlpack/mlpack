/**
 * @file methods/ann/loss_functions/l1_loss.hpp
 * @author Himanshu Pathak
 *
 * Definition of the L1 Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The L1 loss is a loss function that measures the mean absolute error (MAE) 
 * between each element in the input x and target y 
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
class L1Loss
{
 public:
  /**
   * Create the L1Loss object.
   *
   * @param mean Reduction type. If true, it returns the mean of 
   * the loss. Else, it returns the sum.
   */
  L1Loss(const bool mean = true);

  /**
   * Computes the L1 Loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
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

  //! Get the value of reduction type.
  bool Mean() const { return mean; }
  //! Set the value of reduction type.
  bool& Mean() { return mean; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Reduction type. If true, performs mean of loss else sum.
  bool mean;
}; // class L1Loss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "l1_loss_impl.hpp"

#endif
