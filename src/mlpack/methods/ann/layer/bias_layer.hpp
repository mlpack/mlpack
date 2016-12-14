/**
 * @file bias_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BiasLayer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BIAS_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_BIAS_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard bias layer. The BiasLayer class represents a
 * single layer of a neural network.
 *
 * A convenient typedef is given:
 *
 *  - 2DBiasLayer
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
class BiasLayer
{
 public:
  /**
   * Create the BiasLayer object using the specified number of units and bias
   * parameter.
   *
   * @param outSize The number of output units.
   * @param bias The bias value.
   */
  BiasLayer(const size_t outSize, const double bias = 1) :
      outSize(outSize),
      bias(bias)
  {
    weights.set_size(outSize, 1);
  }

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
    output = input + (weights * bias);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = input;
    for (size_t s = 0; s < input.n_slices; s++)
    {
      output.slice(s) += weights(s) * bias;
    }
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
  template<typename DataType, typename ErrorType>
  void Backward(const DataType& /* unused */,
                const ErrorType& gy,
                ErrorType& g)
  {
    g = gy;
  }

  /*
   * Calculate the gradient using the output delta and the bias.
   *
   * @param input The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT, typename ErrorType, typename GradientType>
  void Gradient(const arma::Mat<eT>& /* input */,
                const ErrorType& error,
                GradientType& gradient)
  {
    gradient = error * bias;
  }

  //! Get the weights.
  InputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  InputDataType& Weights() { return weights; }

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
  InputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  InputDataType& Gradient() { return gradient; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(weights, "weights");
    ar & data::CreateNVP(bias, "bias");
  }

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored bias value.
  double bias;

  //! Locally-stored weight object.
  InputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  InputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class BiasLayer

//! Layer traits for the bias layer.
template<typename InputDataType, typename OutputDataType>
class LayerTraits<BiasLayer<InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = true;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

/**
 * Standard 2D-Bias-Layer.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::cube
>
using BiasLayer2D = BiasLayer<InputDataType, OutputDataType>;

/**
 * Standard 2D-Bias-Layer.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using AdditionLayer = BiasLayer<InputDataType, OutputDataType>;

} // namespace ann
} // namespace mlpack

#endif
