/**
 * @file methods/ann/loss_functions/poisson_nll_loss.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the PoissonNLLLossType class. It is the negative log likelihood of
 * the Poisson distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the Poisson negative log likelihood loss. This loss
 * function expects input for each class. It also expects a class index, in the
 * range [0, numClasses - 1], as target when calling the Forward function.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class PoissonNLLLossType
{
 public:
  /**
   * Create the PoissonNLLLossType object.
   *
   * @param logInput If true the loss is computed as
   *        \f$ \exp(input) - target \cdot input \f$, if false then the loss is
   *        \f$ input - target \cdot \log(input + eps) \f$.
   * @param full Boolean value that determines whether to include Stirling's
   *        approximation term.
   * @param eps A small value to prevent 0 in denominators and logarithms.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  PoissonNLLLossType(const bool logInput = true,
                 const bool full = false,
                 const typename MatType::elem_type eps = 1e-08,
                 const bool reduction = true);

  /**
   * Computes the Poisson negative log likelihood Loss.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network. The Poisson Negative Log
   * Likelihood loss function expects the input for each class.
   * It expects a class index, in the range between 1 and the number of classes,
   * as target when calling the Forward function.
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

  //! Get the value of logInput. logInput is a boolean value that tells if
  //! logits are given as input.
  bool LogInput() const { return logInput; }
  //! Modify the value of logInput. logInput is a boolean value that tells if
  //! logits are given as input.
  bool& LogInput() { return logInput; }

  //! Get the value of full. full is a boolean value that determines whether to
  //! include Stirling's approximation term.
  bool Full() const { return full; }
  //! Modify the value of full. full is a boolean value that determines whether
  //! to include Stirling's approximation term.
  bool& Full() { return full; }

  //! Get the value of eps. eps is a small value required to prevent 0 in
  //! logarithms and denominators.
  typename MatType::elem_type Eps() const { return eps; }
  //! Modify the value of eps. eps is a small value required to prevent 0 in
  //! logarithms and denominators.
  typename MatType::elem_type& Eps() { return eps; }

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
  //! Check if the probabilities lie in the range [0, 1].
  void CheckProbs(const MatType& probs)
  {
    for (size_t i = 0; i < probs.size(); ++i)
    {
      if (probs[i] > 1.0 || probs[i] < 0.0)
        Log::Fatal << "Probabilities cannot be greater than 1 "
                   << "or smaller than 0." << std::endl;
    }
  }

  //! Boolean value that tells if logits are given as input.
  bool logInput;

  //! Boolean value that determines whether to include Stirling's
  // approximation term.
  bool full;

  //! eps is a small value required to prevent 0 in logarithms and denominators.
  typename MatType::elem_type eps;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class PoissonNLLLossType

// Default typedef for typical `arma::mat` usage.
using PoissonNLLLoss = PoissonNLLLossType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "poisson_nll_loss_impl.hpp"

#endif
