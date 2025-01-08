/**
 * @file methods/ann/loss_functions/mean_absolute_percentage_error.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Mean Absolute Percentage Error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The mean absolute percentage error performance function measures the
 * network's performance according to the mean of the absolute difference
 * between input and target divided by target.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{de_Myttenaere_2016,
 *    author    = {de Myttenaere, Arnaud and Golden, Boris and Le Grand,
 *                 Bénédicte and Rossi, Fabrice},
 *    title     = {Mean Absolute Percentage Error for regression models},
 *    journal   = {Neurocomputing},
 *    volume    = {abs/1605.02541},
 *    year      = {2016},
 *    url       = {https://arxiv.org/abs/1605.02541},
 *    eprint    = {1605.02541},
 * }
 * @endcode
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class MeanAbsolutePercentageErrorType
{
 public:
  /**
   * Create the MeanAbsolutePercentageErrorType object.
   */
  MeanAbsolutePercentageErrorType();

  /**
   * Computes the mean absolute percentage error function.
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

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) { }
}; // class MeanAbsolutePercentageErrorType

// Default typedef for typical `arma::mat` usage.
using MeanAbsolutePercentageError = MeanAbsolutePercentageErrorType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "mean_absolute_percentage_error_impl.hpp"

#endif
