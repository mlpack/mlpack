/**
 * @file base_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BaseLayer class, which attaches various functions to the
 * embedding layer.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

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
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam DataType Type of data (arma::colvec, arma::mat arma::sp_mat or
 * arma::cube).
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
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

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


}; // namespace ann
}; // namespace mlpack

#endif
