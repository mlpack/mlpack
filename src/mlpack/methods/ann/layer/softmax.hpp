/**
 * @file methods/ann/layer/softmax.hpp
 * @author Mrityunjay Tripathi
 * @author Sreenik Seal
 *
 * Definition of the Softmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMAX_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMAX_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Softmax layer. The softmax function takes as input a
 * vector of K real numbers, and normalizes it into a probability distribution
 * consisting of K probabilities proportional to the exponentials of the input
 * numbers. It should be used for inference only and not with NLL loss (use
 * LogSoftMax instead).
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class Softmax : public Layer<MatType>
{
 public:
  // Convenience typedef to access the element type of the weights and data.
  using ElemType = typename MatType::elem_type;

  // Create the Softmax object.
  Softmax();

  // Clone the Softmax object. This handles polymorphism correctly.
  Softmax* Clone() const { return new Softmax(*this); }

  // Virtual destructor.
  virtual ~Softmax() { }

  // Copy the given Softmax.
  Softmax(const Softmax& other);
  // Take ownership of the given Softmax.
  Softmax(Softmax&& other);
  // Copy the given Softmax.
  Softmax& operator=(const Softmax& other);
  // Take ownership of the given Softmax.
  Softmax& operator=(Softmax&& other);

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

  // Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
}; // class Softmax

} // namespace mlpack

// Include implementation.
#include "softmax_impl.hpp"

#endif
