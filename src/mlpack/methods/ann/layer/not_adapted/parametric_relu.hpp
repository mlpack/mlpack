/**
 * @file methods/ann/layer/parametric_relu.hpp
 * @author Prasanna Patil
 *
 * Definition of PReLU layer first introduced in the,
 * Kaiming He, Xiangyu Zhang, Shaoqing, Ren Jian Sun,
 * "Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification", 2014
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PReLU_HPP
#define MLPACK_METHODS_ANN_LAYER_PReLU_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The PReLU activation function, defined by (where alpha is trainable)
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(x, alpha*x) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *    cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *    to also be in this type. The type also allows the computation and weight
 *    type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class PReLUType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the PReLU object using the specified parameters.
   * The non zero gradient can be adjusted by specifying tha parameter
   * alpha in the range 0 to 1. Default (alpha = 0.03). This parameter
   * is trainable.
   *
   * @param userAlpha Non zero gradient
   */
  PReLUType(const double userAlpha = 0.03);

  //! Clone the PReLUType object. This handles polymorphism correctly.
  PReLUType* Clone() const { return new PReLUType(*this); }

  //! Reset the layer parameter.
  void SetWeights(typename OutputType::elem_type* weightsPtr);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
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
  void Backward(const InputType& input, const OutputType& gy, OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return alpha; }
  //! Modify the parameters.
  OutputType& Parameters() { return alpha; }

  //! Get the non zero gradient.
  double const& Alpha() const { return alpha(0); }
  //! Modify the non zero gradient.
  double& Alpha() { return alpha(0); }

  //! Get size of weights.
  size_t WeightSize() const { return 1; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Leakyness Parameter object.
  OutputType alpha;

  //! Leakyness Parameter given by user in the range 0 < alpha < 1.
  double userAlpha;
}; // class PReLU

// Convenience typedefs.

// Standard PReLU layer.
typedef PReLUType<arma::mat, arma::mat> PReLU;

} // namespace mlpack

// Include implementation.
#include "parametric_relu_impl.hpp"

#endif
