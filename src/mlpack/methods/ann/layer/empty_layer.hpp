/**
 * @file empty_layer.hpp
 * @author Palash Ahuja
 *
 * Definition of the EmptyLayer class, which is basically empty.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_EMPTY_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_EMPTY_LAYER_HPP

namespace mlpack{
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the EmptyLayer class. The EmptyLayer class represents a
 * single layer which is mainly used as placeholder.
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
class EmptyLayer
{
  public:
  /**
   * Creates the empty layer object. All the methods are
   * empty as well.
   */
  EmptyLayer() { /* Nothing to do here. */ }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& /* input */, OutputType& /* output */)
  {
    /* Nothing to do here. */
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
  template<typename InputType, typename ErrorType, typename GradientType>
  void Backward(const InputType& /* input */,
                const ErrorType& /* gy */,
                GradientType& /* g */)
  {
    /* Nothing to do here. */
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename ErrorType, typename GradientType>
  void Gradient(const InputType& /* input */,
                const ErrorType& /* error */,
                GradientType& /* gradient */)
  {
    /* Nothing to do here. */
  }

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }

  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }

  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }

  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }

  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }

  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class EmptyLayer

} //namespace ann
} //namespace mlpack

#endif
