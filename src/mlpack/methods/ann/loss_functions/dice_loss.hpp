/**
 * @file dice_loss.hpp
 * @author N Rajiv Vaidyanathan
 *
 * Definition of the dice loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The dice loss performance function measures the network's
 * performance according to the dice coefficient
 * between the input and target distributions.
 *
 * For more information see the following.
 *
 * @article{Milletari2016,
 *   author    = {Fausto Milletari and Nassir Navab and Seyed{-}Ahmad Ahmadi},
 *   title     = {V-Net: Fully Convolutional Neural Networks for
 *                Volumetric Medical Image Segmentation},
 *   journal   = {CoRR},
 *   volume    = {abs/1606.04797},
 *   year      = {2016},
 *   url       = {http://arxiv.org/abs/1606.04797},
 *   archivePrefix = {arXiv},
 *   eprint    = {1606.04797},
 * }
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
class DiceLoss
{
 public:
  /**
   * Create the DiceLoss object.
   *
   * @param smooth The Laplace smoothing parameter.
   */
  DiceLoss(const double smooth = 1);

  /**
   * Computes the dice loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the smooth.
  double Smooth() const { return smooth; }
  //! Modify the smooth.
  double& Smooth() { return smooth; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The parameter to avoid overfitting.
  double smooth;
}; // class DiceLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "dice_loss_impl.hpp"

#endif
