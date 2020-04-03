/**
 * @file softmax.hpp
 * @author Sreenik Seal
 *
 * Definition of the Softmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMAX_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMAX_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the softmax layer. The softmax loss layer computes
 * the multinomial logistic loss of the softmax of its inputs. It should be
 * used for inference only and not with NLL loss (use logsoftmax instead).
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
class SoftMax
{
 public:
  /**
   * Create the Softmax object.
   */
  SoftMax();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  InputDataType& Delta() const { return delta; }
  //! Modify the delta.
  InputDataType& Delta() { return delta; }

  //! Get the deterministic.
  InputDataType& Deterministic() const { return deterministic; }
  //! Modify the deterministic.
  InputDataType& Deterministic() { return deterministic; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored value of the axis along which softmax is calculated.
  bool axis;

  //! Value of deterministic parameter.
  bool deterministic;

  //! Values from forward pass will be stored in y if non-deterministic.
  arma::Mat<typename InputDataType::elem_type> y;
}; // class SoftMax

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "softmax_impl.hpp"

#endif
