/**
 * @file reconstruction_loss.hpp
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
namespace ann /** Artificial Neural Network. */ {

/**
 * The reconstruction loss performance function measures the network's
 * performance equal to the negative log probability of the target with
 * the input distribution.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam DistType The type of distribution parametrized by the input.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename DistType = BernoulliDistribution<InputDataType>
>
class ReconstructionLoss
{
 public:
  /**
   * Create the ReconstructionLoss object.
   */
  ReconstructionLoss();

  /**
   * Computes the reconstruction loss.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target matrix.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target matrix.
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

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored distribution object.
  DistType dist;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class ReconstructionLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "reconstruction_loss_impl.hpp"

#endif
