/**
 * @file methods/ann/loss_functions/margin_ranking_loss.hpp
 * @author Andrei Mihalea
 *
 * Definition of the Margin Ranking Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ANN_LOSS_FUNCTION_MARGIN_RANKING_LOSS_HPP
#define MLPACK_ANN_LOSS_FUNCTION_MARGIN_RANKING_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Margin ranking loss measures the loss given inputs and a label vector with
 * values of 1 or -1. If the label is 1 then the first input should be ranked
 * higher than the second input at a distance larger than a margin, and vice-
 * versa if the label is -1.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class MarginRankingLossType
{
 public:
  /**
   * Create the MarginRankingLossType object with Hyperparameter margin.
   * Hyperparameter margin defines a minimum distance between correctly ranked
   * samples.
   *
   * @param margin defines a minimum distance between correctly ranked samples.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  MarginRankingLossType(const double margin = 1.0, const bool reduction = true);

  /**
   * Computes the Margin Ranking Loss function.
   * 
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The label vector which contains values of -1 or 1.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The label vector which contains -1 or 1 values.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Get the margin parameter.
  double Margin() const { return margin; }
  //! Modify the margin parameter.
  double& Margin() { return margin; }

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
  //! The margin value used in calculating Margin Ranking Loss.
  double margin;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class MarginRankingLossType

// Default typedef for typical `arma::mat` usage.
using MarginRankingLoss = MarginRankingLossType<arma::mat>;

} // namespace mlpack

// include implementation.
#include "margin_ranking_loss_impl.hpp"

#endif
