/**
 * @file dropconnect_layer.hpp
 * @author Palash Ahuja
 *
 * Definition of the DropConnectLayer class, which implements a regularizer
 * that randomly sets connections to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPCONNECT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPCONNECT_LAYER_HPP

#include <mlpack/core.hpp>

#include "empty_layer.hpp"
#include <mlpack/methods/ann/network_util.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The DropConnect layer is a regularizer that randomly with probability
 * ratio sets the connection values to zero and scales the remaining
 * elements by factor 1 /(1 - ratio). The output is scaled with 1 / (1 - p)
 * when deterministic is false. In the deterministic mode(during testing),
 * the layer just computes the output. The output is computed according
 * to the input layer. If no input layer is given, it will take a linear layer
 * as default.
 *
 * Note:
 * During training you should set deterministic to false and during testing
 * you should set deterministic to true.
 *
 *  For more information, see the following.
 *
 * @code
 * @inproceedings{WanICML2013,
 *   title={Regularization of Neural Networks using DropConnect},
 *   booktitle = {Proceedings of the 30th International Conference on Machine
 *                Learning(ICML - 13)},
 *   author = {Li Wan and Matthew Zeiler and Sixin Zhang and Yann L. Cun and
 *             Rob Fergus},
 *   year = {2013}
 * }
 * @endcode
 *
 * @tparam InputLayer Layer used instead of the internal linear layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
    typename InputLayer = EmptyLayer<arma::mat, arma::mat>,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class DropConnectLayer
{
 public:
 /**
   * Creates the DropConnect Layer as a Linear Object that takes input size,
   * output size and ratio as parameter.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param ratio The probability of setting a value to zero.
   */
  DropConnectLayer (const size_t inSize,
                    const size_t outSize,
                    const double ratio = 0.5) :
      inSize(inSize),
      outSize(outSize),
      ratio(ratio),
      scale(1.0 / (1 - ratio)),
      uselayer(false)
  {
    weights.set_size(outSize, inSize);
  }

  /**
   * Create the DropConnectLayer object using the specified ratio and rescale
   * parameter. This takes the
   *
   * @param ratio The probability of setting a connection to zero.
   * @param inputLayer the layer object that the dropconnect connection would take.
   */
  template<typename InputLayerType>
  DropConnectLayer(InputLayerType &&inputLayer,
                   const double ratio = 0.5) :
      baseLayer(std::forward<InputLayerType>(inputLayer)),
      ratio(ratio),
      scale(1.0 / (1 - ratio)),
      uselayer(true)
  {
    static_assert(std::is_same<typename std::decay<InputLayerType>::type,
                  InputLayer>::value,
                  "The type of the inputLayer must be InputLayerType");
  }
  /**
  * Ordinary feed forward pass of the DropConnect layer.
  *
  * @param input Input data used for evaluating the specified function.
  * @param output Resulting output activation.
  */
  template<typename eT>
  void Forward(const arma::Mat<eT> &input, arma::Mat<eT> &output)
  {
    // The DropConnect mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      if (uselayer)
      {
        baseLayer.Forward(input, output);
      }
      else
      {
        output = weights * input;
      }
    }
    else
    {
      if (uselayer)
      {
        // Scale with input / (1 - ratio) and set values to zero with
        // probability ratio.
        mask = arma::randu<arma::Mat<eT> >(baseLayer.Weights().n_rows,
            baseLayer.Weights().n_cols);
        mask.transform([&](double val) { return (val > ratio); });

        // Save weights for denoising.
        denoise = baseLayer.Weights();

        baseLayer.Weights() = baseLayer.Weights() % mask;

        baseLayer.Forward(input, output);
      }
      else
      {
        // Scale the input / ( 1 - ratio) and set values to zero with
        // probability ratio.
        mask = arma::randu<arma::Mat<eT> >(weights.n_rows, weights.n_cols);
        mask.transform([&](double val) { return (val > ratio); });

        // Save weights for denoising.
        denoise = weights;

        weights = weights % mask;
        output = weights * input;
      }

      output = output * scale;
    }
  }

  /**
   * Ordinary feed backward pass of the DropConnect layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input, const DataType& gy, DataType& g)
  {
    if (uselayer)
    {
      baseLayer.Backward(input, gy, g);
    }
    else
    {
      g = weights.t() * gy;
    }
  }

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The propagated input.
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT, typename GradientDataType>
  void Gradient(const InputType& input,
                const arma::Mat<eT>& d,
                GradientDataType& g)
  {
    if (uselayer)
    {
      baseLayer.Gradient(input, d, g);

      // Denoise the weights.
      baseLayer.Weights() = denoise;
    }
    else
    {
      g = d * input.t();

      // Denoise the weights.
      weights = denoise;
    }
  }

  //! Get the weights.
  OutputDataType const& Weights() const
  {
    if (uselayer)
      return baseLayer.Weights();

    return weights;
  }

  //! Modify the weights.
  OutputDataType& Weights()
  {
    if (uselayer)
      return baseLayer.Weights();

    return weights;
  }

  //! Get the input parameter.
  InputDataType &InputParameter() const
  {
    if (uselayer)
      return baseLayer.InputParameter();

    return inputParameter;
  }

  //! Modify the input parameter.
  InputDataType &InputParameter()
  {
    if (uselayer)
      return baseLayer.InputParameter();

    return inputParameter;
  }

  //! Get the output parameter.
  OutputDataType &OutputParameter() const
  {
    if (uselayer)
      return baseLayer.OutputParameter();

    return outputParameter;
  }

  //! Modify the output parameter.
  OutputDataType &OutputParameter()
  {
    if (uselayer)
      return baseLayer.OutputParameter();

    return outputParameter;
  }

  //! Get the delta.
  OutputDataType const& Delta() const
  {
    if (uselayer)
      return baseLayer.Delta();

    return delta;
  }

  //! Modify the delta.
  OutputDataType& Delta()
  {
    if (uselayer)
      return baseLayer.Delta();

    return delta;
  }

  //! Get the gradient.
  OutputDataType const& Gradient() const
  {
    if (uselayer)
      return baseLayer.Gradient();

    return gradient;
   }

  //! Modify the gradient.
  OutputDataType& Gradient()
  {
    if (uselayer)
      return baseLayer.Gradient();

    return gradient;
  }

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }

  //! Modify the value of the deterministic parameter.
  bool &Deterministic() { return deterministic; }

  //! The probability of setting a value to zero.
  double Ratio() const { return ratio; }

  //! Modify the probability of setting a value to zero.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

private:
  //! Locally-stored layer object.
  InputLayer baseLayer;

  //! Locally stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! If true the default layer is used otherwise a new layer will be created.
  bool uselayer;

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

  //! Locally-stored mast object.
  OutputDataType mask;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Denoise mask for the weights.
  OutputDataType denoise;
}; // class DropConnectLayer.

}  // namespace ann
}  // namespace mlpack

#endif
