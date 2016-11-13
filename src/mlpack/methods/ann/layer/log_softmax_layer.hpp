/**
 * @file log_softmax_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the LogSoftmaxLayer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the log softmax layer. The log softmax loss layer computes
 * the multinomial logistic loss of the softmax of its inputs. This layer is
 * meant to be used in combination with the negative log likelihood layer
 * (NegativeLogLikelihoodLayer), which expects that the input contains
 * log-probabilities for each class.
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
class LogSoftmaxLayer
{
 public:
  /**
   * Create the LogSoftmaxLayer object.
   */
  LogSoftmaxLayer() { /* Nothing to do here. */ }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    arma::mat maxInput = arma::repmat(arma::max(input), input.n_rows, 1);
    output = (maxInput - input);

    // Approximation of the hyperbolic tangent. The acuracy however is
    // about 0.00001 lower as using tanh. Credits go to Leon Bottou.
    output.transform( [](double x)
    {
      //! Fast approximation of exp(-x) for x positive.
      static constexpr double A0 = 1.0;
      static constexpr double A1 = 0.125;
      static constexpr double A2 = 0.0078125;
      static constexpr double A3 = 0.00032552083;
      static constexpr double A4 = 1.0172526e-5;

      if (x < 13.0)
      {
        double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
        y *= y;
        y *= y;
        y *= y;
        y = 1 / y;

        return y;
      }

      return 0.0;
    } );

    output = input - (maxInput + std::log(arma::accu(output)));
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    g = gy - arma::exp(input) * arma::accu(gy);
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
  InputDataType& Delta() const { return delta; }
  //! Modify the delta.
  InputDataType& Delta() { return delta; }

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class LogSoftmaxLayer

}; // namespace ann
}; // namespace mlpack

#endif
