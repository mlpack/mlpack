/**
 * @file methods/ann/layers/hard_swish.hpp
 * @author Anush Kini
 *
 * Definition and implementation of the Hard Swish function as described by
 * Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W, Zhu Y, Pang R,
 * Vasudevan V and Le QV.
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W,
 *            Zhu Y, Pang R, Vasudevan V and Le QV},
 *   title = {Searching for MobileNetV3},
 *   year = {2019}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_SWISH_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_SWISH_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The Hard Swish function, defined by
 *
 * @f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *     0 & x \leq -3\\
 *     x & x \geq +3\\
 *     \frac{x * (x + 3)}{6} & otherwise\\
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *     0 & x \leq -3\\
 *     1 & x \geq +3\\
 *     \frac{2x + 3}{6} & otherwise\\
 *   \end{cases}
 * @f}
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *    cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *    to also be in this type. The type also allows the computation and weight
 *    type to differ from the input type (Default: arma::mat).
 */
template <typename MatType = arma::mat>
class HardSwishType : public Layer<MatType>
{
 public:
  /**
   * Create the HardSwish object using the specified parameters. The range
   * of the linear region can be adjusted by specifying the maxValue and
   * minValue. Default (maxValue = 1, minValue = -1).
   */
  HardSwishType() {};
  
  virtual ~HardSwishType() { }

  //! Copy the other HardSwish layer
  HardSwishType(const HardSwishType& layer);

  //! Take ownership of the members of the other HardSwish Layer
  HardSwishType(HardSwishType&& layer);

  //! Copy the other HardSwish layer
  HardSwishType& operator=(const HardSwishType& layer);

  //! Take ownership of the members of the other HardSwish Layer
  HardSwishType& operator=(HardSwishType&& layer);

  //! Clone the HardSwishType object. This handles polymorphism correctly.
  HardSwishType* Clone() const { return new HardSwishType(*this); }

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
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input, const MatType& gy, MatType& g);

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored first derivative of the activation function.
  MatType derivative;

}; // class HardSwishType

// Convenience typedefs.

// Standard HardSwish layer.
typedef HardSwishType<arma::mat> HardSwish;

} // namespace mlpack

// Include implementation.
#include "hard_swish_impl.hpp"

#endif