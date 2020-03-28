/**
 * @file margin_ranking_loss.hpp
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
namespace ann /** Artificial Neural Network. */ {

/**
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MarginRankingLoss
{
 public:
  /**
   * Create the MarginRankingLoss object with Hyperparameter margin.
   * Hyperparameter margin defines a minimum distance between correctly ranked
   * samples.
   */
  MarginRankingLoss(const double margin = 1.0);

  /**
   * Computes the Margin Ranking Loss function.
   * Measures the loss between two intputs and a label with -1 and 1 values.
   * a value of 1 in the label means the first input should be ranked higher
   * and a value of -1 means the second input should be ranked higher.
   * 
   * @param input Concatenation of the two inputs for evaluating the specified
   * function.
   * @param target The label vector which contains -1 or 1 values.
   */
  template <
    typename InputType,
    typename TargetType
  >
  double Forward(const InputType& input,
                 const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated concatenated input activation.
   * @param target The label vector which contains -1 or 1 values.
   * @param output The calculated error.
   */
  template <
    typename InputType,
    typename TargetType,
    typename OutputType
  >
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the margin parameter.
  double Margin() const { return margin; }
  //! Modify the margin parameter.
  double& Margin() { return margin; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The margin value used in calculating Margin Ranking Loss.
  double margin;
}; // class MarginRankingLoss

} // namespace ann
} // namespace mlpack

// include implementation.
#include "margin_ranking_loss_impl.hpp"

#endif
