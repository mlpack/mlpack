/**
 * @file methods/ann/loss_functions/earth_mover_distance.hpp
 * @author Shikhar Jaiswal
 *
 * Definition of the earth mover distance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The earth mover distance function measures the network's performance
 * according to the Kantorovich-Rubinstein duality approximation.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class EarthMoverDistanceType
{
 public:
  /**
   * Create the EarthMoverDistanceType object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  EarthMoverDistanceType(const bool reduction = true);

  /**
   * Ordinary feed forward pass of a neural network.
   *
   * @param prediction Prediction used for evaluating the specified loss
   *     function.
   * @param target The target vector.
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
  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class EarthMoverDistanceType

// Default typedef for typical `arma::mat` usage.
using EarthMoverDistance = EarthMoverDistanceType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "earth_mover_distance_impl.hpp"

#endif
