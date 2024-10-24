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

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Softmin layer. The Softmin function takes as a input
 * a vector of K real numbers, rescaling them so that the elements of the
 * K-dimensional output vector lie in the range [0, 1] and sum to 1.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class SoftminType : public Layer<MatType>
{
 public:
  //! Create the Softmin object.
  SoftminType();

  //! Clone the SoftminType object. This handles polymorphism correctly.
  SoftminType* Clone() const { return new SoftminType(*this); }

  //! Virtual destructor.
  virtual ~SoftminType() { }

  //! Copy the given SoftminType.
  SoftminType(const SoftminType& other);
  //! Take ownership of the given SoftminType.
  SoftminType(SoftminType&& other);
  //! Copy the given SoftminType.
  SoftminType& operator=(const SoftminType& other);
  //! Take ownership of the given SoftminType.
  SoftminType& operator=(SoftminType&& other);

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
  void Backward(const MatType& /* input */,
                const MatType& output,
                const MatType& gy,
                MatType& g);

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
}; // class SoftminType

// Convenience typedefs.

// Standard Softmin layer using no regularization.
using Softmin = SoftminType<arma::mat>;


} // namespace mlpack

// Include implementation.
#include "softmin_impl.hpp"

#endif
