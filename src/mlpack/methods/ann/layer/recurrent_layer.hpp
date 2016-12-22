/**
 * @file recurrent_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the RecurrentLayer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the RecurrentLayer class. Recurrent layers can be used
 * similarly to feed-forward layers except that the input isn't stored in the
 * inputParameter, instead it's in stored in the recurrentParameter.
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
class RecurrentLayer
{
 public:
  /**
   * Create the RecurrentLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  RecurrentLayer(const size_t inSize, const size_t outSize) :
      inSize(outSize),
      outSize(outSize),
      recurrentParameter(arma::zeros<InputDataType>(inSize, 1))
  {
    weights.set_size(outSize, inSize);
  }

  /**
   * Create the RecurrentLayer object using the specified number of units.
   *
   * @param outSize The number of output units.
   */
  RecurrentLayer(const size_t outSize) :
      inSize(outSize),
      outSize(outSize),
      recurrentParameter(arma::zeros<InputDataType>(outSize, 1))
  {
    weights.set_size(outSize, inSize);
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
    output = input + weights * recurrentParameter;
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
  template<typename InputType, typename eT>
  void Backward(const InputType& /* unused */,
                const arma::Mat<eT>& gy,
                arma::mat& g)
  {
    g = (weights).t() * gy;
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The propagated input activation.
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT, typename GradientDataType>
  void Gradient(const InputType& /* input */,
                const arma::Mat<eT>& d,
                GradientDataType& g)
  {
    g = d * recurrentParameter.t();
  }

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the input parameter.
  InputDataType const& RecurrentParameter() const { return recurrentParameter; }
  //! Modify the input parameter.
  InputDataType& RecurrentParameter() { return recurrentParameter; }

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

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(recurrentParameter, "recurrentParameter");
    ar & data::CreateNVP(weights, "weights");
  }

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

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

  //! Locally-stored recurrent parameter object.
  InputDataType recurrentParameter;
}; // class RecurrentLayer

//! Layer traits for the recurrent layer.
template<typename InputDataType, typename OutputDataType
>
class LayerTraits<RecurrentLayer<InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

} // namespace ann
} // namespace mlpack

#endif
