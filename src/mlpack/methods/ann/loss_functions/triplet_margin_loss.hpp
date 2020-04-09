/**
 * @file triplet_margin_loss.hpp
 * @author Prince Gupta
 *
 * Definition of the Triplet Margin Loss function.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{Schroff2015,
 *   author  = {Florian Schroff, Dmitry Kalenichenko, James Philbin},
 *   title   = {FaceNet: A Unified Embedding for Face Recognition and Clustering},
 *   year    = {2015},
 *   url     = {https://arxiv.org/abs/1503.03832},
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_LOSS_HPP
#define MLPACK_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The TripletMarginLoss function's objective is that the distance from the 
 * anchor input to the positive input is minimized, and the distance from the
 * anchor input to the negative input is maximized.
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
class TripletMarginLoss
{
 public:
  /**
   * Create the TripletMarginLoss object with Hyperparameter margin.
   * Hyperparameter margin defines the minimum value by which the distance
   * between Anchor and Negative sample exceeds the distance between
   * Anchor and Positive sample.
   * The distance between two samples A and B is defined as square of L2 norm
   * of A-B.
   */
  TripletMarginLoss(const double margin = 1.0);

  /**
   * Computes the Triplet Margin Loss function.
   *
   * @param input The propagated input activation. It should be
   *        concatenated anchor and positive samples.
   * @param target The target vector. It should be negative samples.
   */
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType& input,
                                        const TargetType& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation. It should be
   *        concatenated anchor and positive samples.
   * @param target The target vector. It should be negative samples.
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

  //! Get the output parameter.
  double Margin() const { return margin; }
  //! Modify the output parameter.
  double& Margin() { return margin; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The margin value used in calculating Triplet Margin Loss.
  double margin;
}; // class TripletLossMargin

} // namespace ann
} // namespace mlpack

// include implementation.
#include "triplet_margin_loss_impl.hpp"

#endif
