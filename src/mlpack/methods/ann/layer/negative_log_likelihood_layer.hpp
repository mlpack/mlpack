/**
 * @file negative_log_likelihood_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the NegativeLogLikelihoodLayer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_Layer_HPP
#define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_Layer_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the negative log likelihood layer. The negative log
 * likelihood layer expects that the input contains log-probabilities for each
 * class. The layer also expects a class index, in the range between 1 and the
 * number of classes, as target when calling the Forward function.
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class NegativeLogLikelihoodLayer
{
 public:
  /**
   * Create the NegativeLogLikelihoodLayer object.
   */
  NegativeLogLikelihoodLayer() { /* Nothing to do here. */ }

  /**
   * Ordinary feed forward pass of a neural network. The negative log
   * likelihood layer expects that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input Input data that contains the log-probabilities for each class.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  template<typename eT>
  double Forward(const arma::Mat<eT>& input, const arma::Mat<eT>& target)
  {
    double output = 0;

    for (size_t i = 0; i < input.n_cols; ++i)
    {
      size_t currentTarget = target(i) - 1;
      Log::Assert(currentTarget >= 0 && currentTarget < input.n_rows,
          "Target class out of range.");

      output -= input(currentTarget, i);
    }

    return output;
  }

  /**
   * Ordinary feed backward pass of a neural network. The negative log
   * likelihood layer expects that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& target,
                arma::Mat<eT>& output)
  {
    output = arma::zeros<arma::Mat<eT> >(input.n_rows, input.n_cols);
    for (size_t i = 0; i < input.n_cols; ++i)
    {
      size_t currentTarget = target(i) - 1;
      Log::Assert(currentTarget >= 0 && currentTarget < input.n_rows,
          "Target class out of range.");

      output(currentTarget, i) = -1;
    }
  }

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class NegativeLogLikelihoodLayer

}; // namespace ann
}; // namespace mlpack

#endif
