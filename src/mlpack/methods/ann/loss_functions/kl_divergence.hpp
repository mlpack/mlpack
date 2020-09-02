/**
 * @file methods/ann/loss_functions/kl_divergence.hpp
 * @author Dakshit Agrawal
 *
 * Definition of the Kullback–Leibler Divergence error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Kullback–Leibler divergence is often used for continuous
 * distributions (direct regression).
 *
 * For more information, see the following paper.
 *
 * @code
 * article{Kullback1951,
 *   title   = {On Information and Sufficiency},
 *   author  = {S. Kullback, R.A. Leibler},
 *   journal = {The Annals of Mathematical Statistics},
 *   year    = {1951}
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
class KLDivergence
{
 public:
  /**
   * Create the Kullback–Leibler Divergence object with the specified
   * parameters.
   *
   * @param takeMean Boolean variable to specify whether to take mean or not.
   */
  KLDivergence(const bool takeMean = false);

  /**
   * Computes the Kullback–Leibler divergence error function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target Target data to compare with.
   */
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType& input,
                                        const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the value of takeMean.
  bool TakeMean() const { return takeMean; }
  //! Modify the value of takeMean.
  bool& TakeMean() { return takeMean; }

  /**
   * Serialize the loss function
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Boolean variable for taking mean or not.
  bool takeMean;
}; // class KLDivergence

} // namespace ann
} // namespace mlpack

// include implementation
#include "kl_divergence_impl.hpp"

#endif
