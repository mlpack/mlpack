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
namespace ann /** Artificial Neural Network. */ {

/**
 * The earth mover distance function measures the network's performance
 * according to the Kantorovich-Rubinstein duality approximation.
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
class EarthMoverDistance
{
 public:
  /**
   * Create the EarthMoverDistance object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If
   *                  true, 'sum' reduction is used and the output will be
   *                  summed. It is set to true by default.
   */
  EarthMoverDistance(const bool reduction = true);

  /**
   * Ordinary feed forward pass of a neural network.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector, having same shape as input.
   */
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType& input,
                                        const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
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

  //! Get the type of reduction used.
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

    //! The boolean value that tells if reduction is sum or mean.
  bool reduction;
}; // class EarthMoverDistance

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "earth_mover_distance_impl.hpp"

#endif
