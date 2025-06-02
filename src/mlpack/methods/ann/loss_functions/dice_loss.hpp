/**
 * @file methods/ann/loss_functions/dice_loss.hpp
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

/**
 * The dice loss performance function measures the network's
 * performance according to the dice coefficient
 * between the input and target distributions.
 *
 * For more information see the following.
 *
 * @code
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
 * @endcode
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class DiceLossType
{
 public:
  /**
   * Create the DiceLossType object.
   *
   * @param smooth The Laplace smoothing parameter.
   */
  DiceLossType(const double smooth = 1);

  /**
   * Computes the dice loss function.
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

  //! Get the smooth.
  double Smooth() const { return smooth; }
  //! Modify the smooth.
  double& Smooth() { return smooth; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! The parameter to avoid overfitting.
  double smooth;
}; // class DiceLossType

// Default typedef for typical `arma::mat` usage.
using DiceLoss = DiceLossType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "dice_loss_impl.hpp"

#endif
