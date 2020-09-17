/**
 * @file methods/ann/layer/softmin.hpp
 * @author Aakash Kaushik
 *
 * Definition of the Softmin class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMIN_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMIN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Softmin layer. The Softmin function takes as a input
 * a vector of K real numbers, rescaling them so that the elements of the
 * K-dimensional output vector lie in the range [0, 1] and sum to 1.
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
class Softmin
{
 public:
  /**
   * Create the Softmin object.
   */
  Softmin();

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
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
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

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally stored output parameter object.
  OutputDataType outputParameter;
}; // class Softmin

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "softmin_impl.hpp"

#endif
