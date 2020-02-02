/**
 * @file poisson_nll_loss.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the Poisson Negative Log Likelihood class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POISSON_NLL_LOSS_HPP
#define MLPACK_METHODS_ANN_LAYER_POISSON_NLL_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the poisson negative log likelihood layer. The poisson
 * negative log likelihood layer expects input for each class.
 * The layer also expects a class index, in the range between 1 and the
 * number of classes, as target when calling the Forward function.
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
   * @param logInput If true the loss is computed as exp(input) - target*input,
   *        if false then the loss is input - target * log(input + eps)
   * @param full Boolean value that tell if the full loss has to be calculated
   *        i.e to add Stirling approximation term
   * @param eps A small value to prevent log(0) when logInput = False
   * @param reduce Specifies which reduction to be applied to the output
   *        i.e sum'|'mean'. The corresponding boolean values are 0 and 1.
   */
  PoissonNLLLoss(
    const bool logInput = true,
    const bool full = false,
    const double eps = 1e-08,
    const bool reduce = 1);

  /**
   * Computes the Poisson Negative log likelihood.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, TargetType&& target);

  /**
   * Ordinary feed backward pass of a neural network. The poisson negative log
   * likelihood layer expects the input for each class.
   * The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the value of logInput.
  bool logInput() const { return logInput; }
  //! Modify the value of logInput.
  bool& logInput() { return logInput; }

  //! Get the value of full.
  bool full() const { return full; }
  //! Modify the value of full.
  bool& full() { return full; }

  //! Get the value of eps.
  double eps() const { return eps; }
  //! Modify the value of eps.
  double& eps() { return eps; }

  //! Get the value of reduction.
  int reduction() const { return reduction; }
  //! Modify the value of reduction.
  int& reduction() { return reduction; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Small value required to avoid log(0)
  double eps;

  //! Boolean value that tells if input is log(x).
  bool logInput;

  //! Boolean value that tells if full loss has to be calculated.
  bool full;

  //! Boolean value that tells if mean has to be taken.
  bool reduce;
}; // class PoissonNLLLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "poisson_nll_loss_impl.hpp"

#endif
