/**
 * @file methods/ann/layer/hardshrink.hpp
 * @author Lakshya Ojha
 *
 * Same as soft thresholding, if its amplitude is smaller than a predefined
 * threshold, it will be set to zero (kill), otherwise it will be kept
 * unchanged. In order to promote sparsity and to improve the approximation,
 * the hard thresholding method is used as an alternative.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARDSHRINK_HPP
#define MLPACK_METHODS_ANN_LAYER_HARDSHRINK_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Hard Shrink operator is defined as,
 *
 * \f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *     x  & : x >  lambda \\
 *     x  & : x < -lambda \\
 *     0  & : otherwise.
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *     1 & : x >  lambda \\
 *     1 & : x < -lambda \\
 *     0 & : otherwise.
 *   \end{cases}
 * \f}
 *
 * \f$\lambda\f$ is set to 0.5 by default.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class HardShrinkType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create HardShrink object using specified hyperparameter lambda.
   *
   * @param lambda Is calculated by multiplying the noise level sigma of the
   *     input(noisy image) and a coefficient 'a' which is one of the training
   *     parameters. Default value of lambda is 0.5.
   */
  HardShrinkType(const double lambda = 0.5);

  //! Clone the HardShrinkType object. This handles polymorphism correctly.
  HardShrinkType* Clone() const { return new HardShrinkType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the Hard Shrink function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation f(x).
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
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
  //! Locally-stored hyperparameter lambda.
  double lambda;
}; // class HardShrinkType

// Convenience typedefs.

// Standard HardShrink layer.
using HardShrink = HardShrinkType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "hardshrink_impl.hpp"

#endif
