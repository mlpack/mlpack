/**
 * @file methods/ann/loss_functions/reconstruction_loss.hpp
 * @author Atharva Khandait
 *
 * Definition of the reconstruction loss performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_RECONSTRUCTION_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_RECONSTRUCTION_LOSS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>

namespace mlpack {

/**
 * The reconstruction loss performance function measures the network's
 * performance equal to the negative log probability of the target with
 * the input distribution.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @tparam DistType The type of distribution parametrized by the input.
 */
template<
    typename MatType = arma::mat,
    typename DistType = BernoulliDistribution<MatType>
>
class ReconstructionLossType
{
 public:
  /**
   * Create the ReconstructionLossType object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  ReconstructionLossType(const bool reduction = true);

  /**
   * Computes the reconstruction loss.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target matrix.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target matrix.
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
  //! Locally-stored distribution object.
  DistType dist;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class ReconstructionLossType

// Default typedef for typical `arma::mat` usage.
using ReconstructionLoss = ReconstructionLossType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "reconstruction_loss_impl.hpp"

#endif
