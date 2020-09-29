/**
 * @file methods/ann/loss_functions/mean_absolute_percentage_error.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Mean Absolute Percentage Error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The mean absolute percentage error performance function measures the
 * network's performance according to the mean of the absolute difference
 * between input and target divided by target.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{de_Myttenaere_2016,
 *    author    = {de Myttenaere, Arnaud and Golden, Boris and Le Grand,
 *                 Bénédicte and Rossi, Fabrice},
 *    title     = {Mean Absolute Percentage Error for regression models},
 *    journal   = {Neurocomputing},
 *    volume    = {abs/1605.02541},
 *    year      = {2016},
 *    url       = {https://arxiv.org/abs/1605.02541},
 *    eprint    = {1605.02541},
 * }
 * @endcode
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
class MeanAbsolutePercentageError
{
 public:
  /**
   * Create the MeanAbsolutePercentageError object.
   */
  MeanAbsolutePercentageError();

  /**
   * Computes the mean absolute percentage error function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
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

  /**
  * Serialize the layer.
  */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MeanAbsolutePercentageError

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "mean_absolute_percentage_error_impl.hpp"

#endif
