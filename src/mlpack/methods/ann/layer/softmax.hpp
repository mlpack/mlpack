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
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Softmax layer. The softmax function takes as input a
 * vector of K real numbers, and normalizes it into a probability distribution
 * consisting of K probabilities proportional to the exponentials of the input
 * numbers. It should be used for inference only and not with NLL loss (use
 * LogSoftMax instead).
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class SoftmaxType : public Layer<InputType, OutputType>
{
 public:
  //! Create the Softmax object.
  SoftmaxType();

  //! Clone the SoftmaxType object. This handles polymorphism correctly.
  SoftmaxType* Clone() const { return new SoftmaxType(*this); }

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

  //! Get the size of the weights.
  size_t WeightSize() const { return 0; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */);

 private:
}; // class SoftmaxType

// Convenience typedefs.

// Standard Linear layer using no regularization.
typedef SoftmaxType<arma::mat, arma::mat> Softmax;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "softmax_impl.hpp"

#endif
