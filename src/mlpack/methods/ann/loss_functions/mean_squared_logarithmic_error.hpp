/**
 * @file methods/ann/loss_functions/mean_squared_logarithmic_error.hpp
 * @author Saksham Rastogi
 *
 * Definition of the mean squared logarithmic error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_LOGARITHMIC_ERROR_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_LOGARITHMIC_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The mean squared logarithmic error performance function measures the
 * network's performance according to the mean of squared logarithmic errors.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class MeanSquaredLogarithmicErrorType
{
 public:
  /**
   * Create the MeanSquaredLogarithmicErrorType object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  MeanSquaredLogarithmicErrorType(const bool reduction = true);

  /**
   * Computes the mean squared logarithmic error function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
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
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class MeanSquaredLogarithmicErrorType

// Default typedef for typical `arma::mat` usage.
using MeanSquaredLogarithmicError =
    MeanSquaredLogarithmicErrorType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "mean_squared_logarithmic_error_impl.hpp"

#endif
