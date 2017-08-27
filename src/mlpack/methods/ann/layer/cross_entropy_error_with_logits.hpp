/**
 * @file cross_entropy_error_with_logits.hpp
 * @author Kris Singh
 *
 * Definition of the cross-entropy with logit performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_LOGIT_ERROR_HPP
#define MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_LOGIT_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The cross-entropy with logits performance function measures the network's
 * performance according to the cross-entropy function.
 * between the input and target distributions.
 * For more detail look here goo.gl/tRjS6j
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class CrossEntropyErrorLogits
{
 public:
  /**
   * Create the CrossEntropyErrorLogits object.
   * 
   * @param eps The minimum value used for computing logarithms
   *            and denominators in a numerically stable way.
   */
  CrossEntropyErrorLogits();

  /*
   * Computes the cross-entropy with logits function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  double Forward(const arma::Mat<eT>&& input, const arma::Mat<eT>&& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                const arma::Mat<eT>&& target,
                arma::Mat<eT>&& output);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class CrossEntropyErrorLogits

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "cross_entropy_error_with_logits_impl.hpp"

#endif
