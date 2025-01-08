/**
 * @file methods/ann/loss_functions/negative_log_likelihood.hpp
 * @author Marcus Edel
 *
 * Definition of the NegativeLogLikelihoodType class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_HPP
#define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the negative log likelihood layer. The negative log
 * likelihood layer expects that the input contains log-probabilities for each
 * class. The layer also expects a class index in the range [0, numClasses - 1]
 * number of classes, as target when calling the Forward function.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class NegativeLogLikelihoodType
{
 public:
  /**
   * Create the NegativeLogLikelihoodTypeLayer object.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  NegativeLogLikelihoodType(const bool reduction = true);

  /**
   * Computes the Negative log likelihood.
   *
   * @param iprediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  double Forward(const MatType& prediction,
                 const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network. The negative log
   * likelihood layer expects that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
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
  void serialize(Archive& /* ar */, const uint32_t /* version */);

 private:
  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class NegativeLogLikelihoodType

// Default typedef for typical `arma::mat` usage.
using NegativeLogLikelihood = NegativeLogLikelihoodType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "negative_log_likelihood_impl.hpp"

#endif
