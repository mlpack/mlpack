/**
 * @file base_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BaseLayer class, which attaches various functions to the
 * embedding layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the base layer. The base layer works as a metaclass which
 * attaches various functions to the embedding layer.
 *
 * A few convenience typedefs are given:
 *
 *  - SigmoidLayer
 *  - IdentityLayer
 *  - ReLULayer
 *  - TanHLayer
 *  - BaseLayer2D
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class BaseLayer
{
 public:
  /**
   * Create the BaseLayer object.
   */
  BaseLayer()
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output)
  {
    ActivationFunction::fn(input, output);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input,
                const DataType& gy,
                DataType& g)
  {
    DataType derivative;
    ActivationFunction::deriv(input, derivative);
    g = gy % derivative;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Cube<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Cube<eT>& g)
  {
    // Generate a cube using the backpropagated error matrix.
    arma::Cube<eT> mappedError = arma::zeros<arma::cube>(input.n_rows,
        input.n_cols, input.n_slices);

    for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= gy.n_cols, j++)
    {
      for (size_t i = 0; i < gy.n_cols; i++)
      {
        arma::Col<eT> temp = gy.col(i).subvec(
            j * input.n_rows * input.n_cols,
            (j + 1) * input.n_rows * input.n_cols - 1);

        mappedError.slice(s + i) = arma::Mat<eT>(temp.memptr(),
            input.n_rows, input.n_cols);
      }
    }

    arma::Cube<eT> derivative;
    ActivationFunction::deriv(input, derivative);
    g = mappedError % derivative;
  }

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

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */)
  {
    /* Nothing to do here */
  }

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class BaseLayer

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using SigmoidLayer = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;

/**
 * Standard Identity-Layer using the identity activation function.
 */
template <
    class ActivationFunction = IdentityFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using IdentityLayer = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;

/**
 * Standard rectified linear unit non-linearity layer.
 */
template <
    class ActivationFunction = RectifierFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using ReLULayer = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;

/**
 * Standard hyperbolic tangent layer.
 */
template <
    class ActivationFunction = TanhFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using TanHLayer = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;

/**
 * Standard Base-Layer2D using the logistic activation function.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::cube
>
using BaseLayer2D = BaseLayer<
    ActivationFunction, InputDataType, OutputDataType>;


} // namespace ann
} // namespace mlpack

#endif
