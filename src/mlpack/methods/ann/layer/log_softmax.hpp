/**
 * @file methods/ann/layer/log_softmax.hpp
 * @author Marcus Edel
 *
 * Definition of the LogSoftmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_HPP
#define MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the log softmax layer. The log softmax loss layer computes
 * the multinomial logistic loss of the softmax of its inputs. This layer is
 * meant to be used in combination with the negative log likelihood layer
 * (NegativeLogLikelihoodLayer), which expects that the input contains
 * log-probabilities for each class.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <typename MatType = arma::mat>
class LogSoftMaxType : public Layer<MatType>
{
 public:
  /**
   * Create the LogSoftmax layer.
   */
  LogSoftMaxType();

  //! Clone the LogSoftMaxType object. This handles polymorphism correctly.
  LogSoftMaxType* Clone() const { return new LogSoftMaxType(*this); }

  // Virtual destructor.
  virtual ~LogSoftMaxType() { }

  //! Copy the given LogSoftMaxType.
  LogSoftMaxType(const LogSoftMaxType& other);
  //! Take ownership of the given LogSoftMaxType.
  LogSoftMaxType(LogSoftMaxType&& other);
  //! Copy the given LogSoftMaxType.
  LogSoftMaxType& operator=(const LogSoftMaxType& other);
  //! Take ownership of the given LogSoftMaxType.
  LogSoftMaxType& operator=(LogSoftMaxType&& other);

  /**
   * A wrapper function to call the correct implementation according to the
   * specific matrix type (e.g., arma, coot).
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void ForwardImpl(const MatType& input, MatType& output,
                   const typename std::enable_if_t<
                       arma::is_arma_type<MatType>::value>* = 0);

#ifdef MLPACK_HAS_COOT
  void ForwardImpl(const MatType& input, MatType& output,
                   const typename std::enable_if_t<
                       coot::is_coot_type<MatType>::value>* = 0);
#endif

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
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

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<Layer<MatType>>(this));
    // Nothing to do.
  }
}; // class LogSoftmaxType

// Convenience typedefs.

// Standard Linear layer using no regularization.
using LogSoftMax = LogSoftMaxType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "log_softmax_impl.hpp"

#endif
