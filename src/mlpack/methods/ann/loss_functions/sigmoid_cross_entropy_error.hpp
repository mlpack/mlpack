/**
 * @file methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * Definition of the cross-entropy with logit performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The SigmoidCrossEntropyError performance function measures the network's
 * performance according to the cross-entropy function between the input and
 * target distributions. This function calculates the cross entropy
 * given the real values instead of providing the sigmoid activations.
 * The function uses this equivalent formulation:
 * \f$max(x, 0) - x * z + \log(1 +  e^{-|x|})\f$
 * where x = input and z = target.
 *
 * For more information, see the following paper.
 *
 * @code
 * @article{Janocha2017
 *   title   = {On Loss Functions for Deep Neural Networks in Classification},
 *   author  = {Katarzyna Janocha, Wojciech Marian Czarnecki},
 *   url     = {http://arxiv.org/abs/1702.05659},
 *   journal = {CoRR},
 *   eprint  = {arXiv:1702.05659},
 *   year    = {2017}
 * }
 * @endcode
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
class SigmoidCrossEntropyError
{
 public:
  /**
   * Create the SigmoidCrossEntropyError object.
   */
  SigmoidCrossEntropyError();

  /**
   * Computes the Sigmoid CrossEntropy Error functions.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
   */
  template<typename InputType, typename TargetType>
  inline typename InputType::elem_type Forward(const InputType& input,
                                               const TargetType& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  inline void Backward(const InputType& input,
                       const TargetType& target,
                       OutputType& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class SigmoidCrossEntropy

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "sigmoid_cross_entropy_error_impl.hpp"

#endif
