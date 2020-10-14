/**
 * @file methods/ann/loss_functions/poisson_nll_loss.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the PoissonNLLLoss class. It is the negative log likelihood of
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
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Poisson negative log likelihood loss. This loss
 * function expects input for each class. It also expects a class index,
 * in the range between 1 and the number of classes, as target when calling
 * the Forward function.
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
class PoissonNLLLoss
{
 public:
  /**
   * Create the PoissonNLLLoss object.
   *
   * @param logInput If true the loss is computed as
   *        \f$ \exp(input) - target \cdot input \f$, if false then the loss is
   *        \f$ input - target \cdot \log(input + eps) \f$.
   * @param full Boolean value that determines whether to include Stirling's
   *        approximation term.
   * @param eps A small value to prevent 0 in denominators and logarithms.
   * @param mean When true, mean loss is computed otherwise total loss.
   */
  PoissonNLLLoss(const bool logInput = true,
                 const bool full = false,
                 const typename InputDataType::elem_type eps = 1e-08,
                 const bool mean = true);

  /**
   * Computes the Poisson negative log likelihood Loss.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  template<typename InputType, typename TargetType>
  typename InputDataType::elem_type Forward(const InputType& input,
                                            const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network. The Poisson Negative Log
   * Likelihood loss function expects the input for each class.
   * It expects a class index, in the range between 1 and the number of classes,
   * as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

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
  typename InputDataType::elem_type Eps() const { return eps; }
  //! Modify the value of eps. eps is a small value required to prevent 0 in
  //! logarithms and denominators.
  typename InputDataType::elem_type& Eps() { return eps; }

  //! Get the value of mean. It's a boolean value that tells if
  //! mean of the total loss has to be taken.
  bool Mean() const { return mean; }
  //! Modify the value of mean. It's a boolean value that tells if
  //! mean of the total loss has to be taken.
  bool& Mean() { return mean; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Check if the probabilities lie in the range [0, 1].
  template<typename eT>
  void CheckProbs(const arma::Mat<eT>& probs)
  {
    for (size_t i = 0; i < probs.size(); ++i)
    {
      if (probs[i] > 1.0 || probs[i] < 0.0)
        Log::Fatal << "Probabilities cannot be greater than 1 "
                   << "or smaller than 0." << std::endl;
    }
  }

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Boolean value that tells if logits are given as input.
  bool logInput;

  //! Boolean value that determines whether to include Stirling's
  // approximation term.
  bool full;

  //! eps is a small value required to prevent 0 in logarithms and denominators.
  typename InputDataType::elem_type eps;

  //! Boolean value that tells if mean of the total loss has to be taken.
  bool mean;
}; // class PoissonNLLLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "poisson_nll_loss_impl.hpp"

#endif
