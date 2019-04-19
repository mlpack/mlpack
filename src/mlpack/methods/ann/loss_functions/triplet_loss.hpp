/**
 * @file triplet_loss.hpp
 * @author Hemal Mamtora
 *
 * Definition of the triplet loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_TRIPLET_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_TRIPLET_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Triplet Loss minimizes the distance between an anchor and a positive,
 * both of which have the same identity, and maximizes the distance between 
 * the anchor and a negative of a different identity.
 *
 * For more information refer to the following paper:
 * https://arxiv.org/pdf/1503.03832.pdf
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
class TripletLoss{
 public:
  /**
   * Create the TripletLoss object.
   *
   * @param margin 
   */
  TripletLoss(const double margin = 0.2);

  /**
   * Computes the triplet loss function.
   *
   * @param anchor Anchor with which postive and negative example will 
   * be compared.
   * @param positive Example which belongs to same class as anchor.
   * @param negative Example which belongs to different class than the anchor
   * @param target The target vector.
   */
  template<typename InputType>
  double Forward(const InputType&& anchor, 
                 const InputType&& positive,
                 const InputType&& negative); 

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param anchor Anchor with which postive and negative example will 
   * be compared.
   * @param positive Example which belongs to same class as anchor.
   * @param negative Example which belongs to different class than the anchor
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename OutputType>
  void Backward(const InputType&& anchor, 
                const InputType&& positive,
                const InputType&& negative,
                OutputType&& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the margin.
  double Margin() const { return margin; }
  //! Modify the margin.
  double& Margin() { return margin; }

  /**
  * Serialize the layer.
  */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The margin parameter in the loss function
  double margin;
}; // class TripletLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "triplet_loss_impl.hpp"

#endif

