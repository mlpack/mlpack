/**
 * @file methods/ann/layer/flatten_t_swish.hpp
 * @author Fawwaz Mayda
 *
 * Definition of Flatten T Swish layer first introduced in the acoustic model,
 * Hock Hung Chieng, Noorhaniza Wahid, Pauline Ong, Sai Raj Kishore Perla,
 * "Flatten-T Swish: a thresholded ReLU-Swish-like activation function for deep learning", 2018
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FLATTEN_T_SWISH_HPP
#define MLPACK_METHODS_ANN_LAYER_FLATTEN_T_SWISH_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Flatten T Swish activation function, defined by
 *
 * @f{eqnarray*}{
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     frac{x}{1+exp(-x)} + T & : x \ge 0 \\
 *     T & : x < 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     \sigma(x)(1 - f(x)) + f(x) & : x > 0 \\
 *     0 & : x \le 0
 *   \end{array}
 * \right.
 * @f}
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
class FlattenTSwish
{
 public:
  /**
   * Create the Flatten T Swish object using the specified parameters.
   * The thresholded value T can be adjusted via T paramaters.
   * When the x is < 0, T will be used instead of 0.
   * The default value of T is -0.20 as suggested in the paper.
   * @param T 
   */
  FlattenTSwish(const double T = -0.20);

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
  template<typename DataType>
  void Backward(const DataType& input, const DataType& gy, DataType& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the T parameter.
  double const& T() const { return t; }
  //! Modify the T parameter.
  double& T() { return t; }

  //! Get size of weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! T Parameter from paper.
  double t;
}; // class FlattenTSwish

} // namespace mlpack

// Include implementation.
#include "flatten_t_swish_impl.hpp"

#endif
