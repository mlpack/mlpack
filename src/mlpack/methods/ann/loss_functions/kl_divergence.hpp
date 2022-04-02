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
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class KLDivergenceType
{
 public:
  /**
   * Create the Kullback–Leibler Divergence object with the specified
   * parameters.
   *
   * @param takeMean Boolean variable to specify whether to take mean or not.
   */
  KLDivergenceType(const bool takeMean = false);

  /**
   * Computes the Kullback–Leibler divergence error function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target Target data to compare with.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Get the value of takeMean.
  bool TakeMean() const { return takeMean; }
  //! Modify the value of takeMean.
  bool& TakeMean() { return takeMean; }

  /**
   * Serialize the loss function
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Boolean variable for taking mean or not.
  bool takeMean;
}; // class KLDivergenceType

// Default typedef for typical `arma::mat` usage.
typedef KLDivergenceType<arma::mat> KLDivergence;

} // namespace ann
} // namespace mlpack

// include implementation
#include "kl_divergence_impl.hpp"

#endif
