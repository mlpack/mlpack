/**
 * @file cross_entropy_error.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the cross-entropy performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_ERROR_HPP
#define MLPACK_METHODS_ANN_LAYER_CROSS_ENTROPY_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The cross-entropy performance function measures the network's
 * performance according to the cross-entropy
 * between the input and target distributions.
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
class CrossEntropyError
{
 public:
  /**
   * Create the CrossEntropyError object.
   *
   * @param eps The minimum value used for computing logarithms
   *            and denominators in a numerically stable way.
   */
  CrossEntropyError(double eps = 1e-10);

  /*
   * Computes the cross-entropy function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

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

  //! Get the epsilon.
  double Eps() const { return eps; }
  //! Modify the epsilon.
  double& Eps() { return eps; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The minimum value used for computing logarithms and denominators
  double eps;
}; // class CrossEntropyError

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "cross_entropy_error_impl.hpp"

#endif
