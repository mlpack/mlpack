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
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  KLDivergenceType(const bool reduction = true);

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

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the loss function.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class KLDivergenceType

// Default typedef for typical `arma::mat` usage.
using KLDivergence = KLDivergenceType<arma::mat>;

} // namespace mlpack

// include implementation
#include "kl_divergence_impl.hpp"

#endif
