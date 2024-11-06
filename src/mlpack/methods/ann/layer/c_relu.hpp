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

#include "layer.hpp"

namespace mlpack {

/**
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
 * @tparam MatType Matrix representation to accept as input and allows the
 *         computation and weight type to differ from the input type
 *         (Default: arma::mat).
 */
template<typename MatType = arma::mat>
class CReLUType : public Layer<MatType>
{
 public:
  //! Create the CReLU object.
  CReLUType();

  //! Clone the CReLUType object. This handles polymorphism correctly.
  CReLUType* Clone() const { return new CReLUType(*this); }

  // Virtual destructor.
  virtual ~CReLUType() { }

  //! Copy the given CReLUType.
  CReLUType(const CReLUType& other);
  //! Take ownership of the given CReLUType.
  CReLUType(CReLUType&& other);
  //! Copy the given CReLUType.
  CReLUType& operator=(const CReLUType& other);
  //! Take ownership of the given CReLUType.
  CReLUType& operator=(CReLUType&& other);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * Works only for 2D Tensors.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  //! Compute the output dimensions of the layer using `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */);
}; // class CReLUType

// Convenience typedefs.

// Standard CReLU layer.
using CReLU = CReLUType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "c_relu_impl.hpp"

#endif
