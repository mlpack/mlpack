/**
 * @file methods/ann/layer/flatten_t_swish.hpp
 * @author Mayank Raj
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
#ifndef MLPACK_METHODS_ANN_LAYER_FTSWISH_HPP
#define MLPACK_METHODS_ANN_LAYER_FTSWISH_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

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
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class FTSwishType : public Layer<MatType>
{
 public:
  /**
   * Create the Flatten T Swish object using the specified parameters.
   * The thresholded value T can be adjusted via T paramaters.
   * When the x is < 0, T will be used instead of 0.
   * The default value of T is -0.20 as suggested in the paper.
   * @param T 
   */
  FTSwishType(const double T = -0.20);

  //! Clone the FTSwishType object. This handles polymorphism correctly.
  FTSwishType* Clone() const { return new FTSwishType(*this); }

  // Virtual destructor.
  virtual ~FTSwishType() { }

  //! Copy the given FTSwishType.
  FTSwishType(const FTSwishType& other);
  //! Take ownership of the given FTSwishType.
  FTSwishType(FTSwishType&& other);
  //! Copy the given FTSwishType.
  FTSwishType& operator=(const FTSwishType& other);
  //! Take ownership of the given FTSwishType.
  FTSwishType& operator=(FTSwishType&& other);

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
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  //! Get the threshold value.
  double const& Threshold() const { return T; }
  //! Modify the threshold value.
  double& Threshold() { return T; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Threshold value for x < 0.
  double T;
}; // class FTSwishType

// Convenience typedefs.
using FTSwish = FTSwishType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "ftswish_impl.hpp"

#endif
