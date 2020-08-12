/**
 * @file methods/ann/layer/c_relu.hpp
 * @author Jeffin Sam
 *
 * Implementation of CReLU layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_C_RELU_HPP
#define MLPACK_METHODS_ANN_LAYER_C_RELU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {
/**
 *
 * A concatenated ReLU has two outputs, one ReLU and one negative ReLU,
 * concatenated together. In other words, for positive x it produces [x, 0],
 * and for negative x it produces [0, x]. Because it has two outputs,
 * CReLU doubles the output dimension.
 *
 * Note:
 * The CReLU doubles the output size.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{ICML2016,
 *   title  = {Understanding and Improving Convolutional Neural Networks
 *             via Concatenated Rectified Linear Units},
 *   author = {LWenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee},
 *   year   = {2016},
 *   url    = {https://arxiv.org/abs/1603.05201}
 * }
 * @endcode
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
class CReLU
{
 public:
  /**
   * Create the CReLU object.
   */
  CReLU();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * Works only for 2D Tensors.
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
}; // class CReLU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "c_relu_impl.hpp"

#endif
