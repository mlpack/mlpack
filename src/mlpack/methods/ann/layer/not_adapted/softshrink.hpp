/**
 * @file methods/ann/layer/softshrink.hpp
 * @author Lakshya Ojha
 *
 * The soft shrink function has threshold proportional to the noise level given
 * by the user. The use of a Soft Shrink activation function provides adaptive
 * denoising at various noise levels using a single
 * CNN (Convolution Neural Network) without a requirement to train a unique CNN
 * for each noise level.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Soft Shrink operator is defined as,
 * \f{eqnarray*}{
 *   f(x) &=&
 *   \begin{cases}
 *     x - \lambda & : x >  \lambda \\
 *     x + \lambda & : x < -\lambda \\
 *     0 & : otherwise. \\
 *   \end{cases} \\
 *   f'(x) &=&
 *   \begin{cases}
 *     1 & : x >  \lambda \\
 *     1 & : x < -\lambda \\
 *     0 & : otherwise.
 *   \end{cases}
 * \f}
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class SoftShrinkType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create SoftShrink object using specified hyperparameter lambda.
   *
   * @param lambda The noise level of an image depends on settings of an
   *     imaging device. The settings can be used to select appropriate
   *     parameters for denoising methods. It is proportional to the noise
   *     level entered by the user. And it is calculated by multiplying the
   *     noise level sigma of the input(noisy image) and a coefficient 'a'
   *     which is one of the training parameters. Default value of lambda
   *     is 0.5.
   */
  SoftShrinkType(const double lambda = 0.5);

  //! Clone the SoftShrinkType object. This handles polymorphism correctly.
  SoftShrinkType* Clone() const { return new SoftShrinkType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the Soft Shrink function.
   * @param output Resulting output activation
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation f(x).
   * @param gy The backpropagated error.
   * @param g The calculated gradient
   */
  void Backward(const InputType& input, const OutputType& gy, OutputType& g);

  //! Get the hyperparameter lambda.
  double const& Lambda() const { return lambda; }
  //! Modify the hyperparameter lambda.
  double& Lambda() { return lambda; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored hyperparamater lambda.
  double lambda;
}; // class SoftShrinkType

// Convenience typedefs.

// Standard SoftShrink layer.
using SoftShrink = SoftShrinkType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "softshrink_impl.hpp"

#endif
