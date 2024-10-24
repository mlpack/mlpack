/**
 * @file methods/ann/layer/flexible_relu.hpp
 * @author Aarush Gupta
 * @author Manthan-R-Sheth
 *
 * Definition of the FlexibleReLU layer as described by Suo Qiu, Xiangmin Xu and
 * Bolun Cai in "FReLU: Flexible Rectified Linear Units for Improving
 * Convolutional Neural Networks".
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLE_RELU_HPP
#define MLPACK_METHODS_ANN_LAYER_FLEXIBLE_RELU_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The FlexibleReLU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(0,x)+alpha \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *      1 & : x > 0 \\
 *      0 & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{Qiu2018,
 *  author  = {Suo Qiu, Xiangmin Xu and Bolun Cai},
 *  title   = {FReLU: Flexible Rectified Linear Units for Improving
 *             Convolutional Neural Networks}
 *  journal = {arxiv preprint},
 *  URL     = {https://arxiv.org/abs/1706.08098},
 *  year    = {2018}
 * }
 * @endcode
 *
 * @tparam MatType Matrix representation to accept as input and allows the
 *         computation and weight type to differ from the input type
 *         (Default: arma::mat).
 */
template<typename MatType = arma::mat>
class FlexibleReLUType : public Layer<MatType>
{
 public:
  /**
   * Create the FlexibleReLU object using the specified alpha parameter.
   * The trainable alpha parameter controls the range of the ReLU function.
   * (Default alpha = 0).
   *
   * @param alpha Parameter to adjust the range of the ReLU function.
   */
  FlexibleReLUType(const double alpha = 0);

  //! Clone the FlexibleReLUType object. This handles polymorphism correctly.
  FlexibleReLUType* Clone() const { return new FlexibleReLUType(*this); }

  // Virtual destructor.
  virtual ~FlexibleReLUType() { }

  //! Copy the given FlexibleReLUType.
  FlexibleReLUType(const FlexibleReLUType& other);
  //! Take ownership of the given FlexibleReLUType.
  FlexibleReLUType(FlexibleReLUType&& other);
  //! Copy the given FlexibleReLUType.
  FlexibleReLUType& operator=(const FlexibleReLUType& other);
  //! Take ownership of the given FlexibleReLUType.
  FlexibleReLUType& operator=(FlexibleReLUType&& other);

  /**
   * Reset the layer parameter (alpha). The method is called to
   * assign the allocated memory to the learnable layer parameter.
   */
  void SetWeights(const MatType& weightsIn);

  /**
   * Initialize the weight matrix of the layer.
   *
   * @param W Weight matrix to initialize.
   * @param elements Number of elements.
   */
  void CustomInitialize(
      MatType& W,
      const size_t elements);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
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
                const MatType& output,
                const MatType& gy,
                MatType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  MatType const& Parameters() const { return alpha; }
  //! Modify the parameters.
  MatType& Parameters() { return alpha; }

  //! Get the parameter controlling the range of the ReLU function.
  const double& Alpha() const { return alpha; }
  //! Modify the parameter controlling the range of the ReLU function.
  double& Alpha() { return alpha; }

  //! Get size of weights.
  size_t WeightSize() const { return 1; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version*/);

 private:
  //! Parameter object.
  MatType alpha;

  //! Parameter controlling the range of the ReLU function.
  double userAlpha;
}; // class FlexibleReLUType

// Convenience typedefs.

// Standard flexible ReLU layer.
using FlexibleReLU = FlexibleReLUType<arma::mat>;

} // namespace mlpack

// Include implementation
#include "flexible_relu_impl.hpp"

#endif
