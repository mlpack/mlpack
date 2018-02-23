/**
 * @file policy.hpp
 * @author Chirag Pabbaraju
 *
 * Definition of the Policy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POLICY_HPP
#define MLPACK_METHODS_ANN_LAYER_POLICY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Policy layer, which implements the policy gradient
 * algorithm. This layer computes the softmax of an input in forward pass, and gradients wrt softmax layer in the backward pass.
 * This layer is meant to be used as the last layer in the neural network predicting the policy.
 * 
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat
>
class Policy
{
 public:
  /**
   * Create the Policy object.
   */
  Policy();

  /**
   * Forward pass of the layer.
   * Computes softmax values of the input.
   *
   * @param input Input data to be converted to softmax values.
   * @param output Resulting output softmax values.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output);

  /**
   * Backward pass of the layer.
   * Computes the gradients of the advantage * log (probablity) of action.
   *
   * @param input The propagated input activation.
   * @param gy The advantage values to backpropogate against.
   * @param g The calculated gradients.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& prob,
                arma::Mat<eT>&& advanatage,
                arma::Mat<eT>&& g);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  InputDataType& Delta() const { return delta; }
  //! Modify the delta.
  InputDataType& Delta() { return delta; }

 private:

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Policy

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "policy_impl.hpp"

#endif