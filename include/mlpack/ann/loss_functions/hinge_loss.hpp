/**
 * @file methods/ann/loss_functions/hinge_loss.hpp
 * @author Anush Kini
 *
 * Definition of the Hinge Loss Function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Computes the hinge loss between \f$y_true\f$ and \f$y_pred\f$. Expects
 * \f$y_true\f$ to be either -1 or 1. If \f$y_true\f$ is either 0 or 1, a
 * temporary conversion is made to calculate the loss.
 * The hinge loss \f$l(y_true, y_pred)\f$ is defined as
 * \f$l(y_true, y_pred) = max(0, 1 - y_true*y_pred)\f$.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class HingeLossType
{
 public:
  /**
   * Create HingeLossType object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If
   *                  true, 'sum' reduction is used and the output will be
   *                  summed. It is set to true by default.
   */
  HingeLossType(const bool reduction = true);

  /**
   * Computes the Hinge loss function.
   *
   * @param prediction Prediction used for evaluating the specified loss
   *     function.
   * @param target Target data to compare with.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Prediction used for evaluating the specified loss
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
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! The boolean value that tells if reduction is sum or mean.
  bool reduction;
}; // class HingeLossType

// Default typedef for typical `arma::mat` usage.
typedef HingeLossType<arma::mat> HingeLoss;

} // namespace mlpack

// include implementation
#include "hinge_loss_impl.hpp"

#endif
