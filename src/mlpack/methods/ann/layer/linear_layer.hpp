/**
 * @file linear_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the LinearLayer class also known as fully-connected layer or
 * affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the LinearLayer class. The LinearLayer class represents a
 * single layer of a neural network.
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
class LinearLayer
{
 public:
  /**
   * Create the LinearLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  LinearLayer(const size_t inSize, const size_t outSize) :
      inSize(inSize),
      outSize(outSize)
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
    output = weights * input;
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Mat<eT>& output)
  {
    arma::Mat<eT> data(input.n_elem, 1);

    for (size_t s = 0, c = 0; s < input.n_slices / data.n_cols; s++)
    {
      for (size_t i = 0; i < data.n_cols; i++, c++)
      {
        data.col(i).subvec(s * input.n_rows * input.n_cols, (s + 1) *
            input.n_rows * input.n_cols - 1) = arma::trans(arma::vectorise(
            input.slice(c), 1));
      }
    }

    output = weights * data;
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
                arma::Mat<eT>& g)
  {
    g = weights.t() * gy;
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename InputType, typename ErrorType, typename GradientType>
  void Gradient(const InputType& input,
                const ErrorType& error,
                GradientType& gradient)
  {
    GradientDelta(input, error, gradient);
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

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(weights, "weights");
  }

 private:
  /*
   * Calculate the gradient using the output delta (3rd order tensor) and the
   * input activation (3rd order tensor).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Cube<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Cube<eT>& g)
  {
    g = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    arma::Mat<eT> data = arma::Mat<eT>(d.n_cols,
        input.n_elem / d.n_cols);

    for (size_t s = 0, c = 0; s < input.n_slices /
        data.n_rows; s++)
    {
      for (size_t i = 0; i < data.n_rows; i++, c++)
      {
        data.row(i).subvec(s * input.n_rows *
            input.n_cols, (s + 1) *
            input.n_rows *
        input.n_cols - 1) = arma::vectorise(
                input.slice(c), 1);
      }
    }

    g.slice(0) = d * data / d.n_cols;
  }

  /*
   * Calculate the gradient (3rd order tensor) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Mat<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Cube<eT>& g)
  {
    g = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(input, d, g.slice(0));
  }

  /*
   * Calculate the gradient (dense matrix) using the output delta
   * (dense matrix) and the input activation (3rd order tensor).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Cube<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Mat<eT>& g)
  {
    arma::Cube<eT> grad = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(input, d, grad);
    g = grad.slice(0);
  }

  /*
   * Calculate the gradient (dense matrix) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Mat<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Mat<eT>& g)
  {
    g = d * input.t();
  }

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
}; // class LinearLayer

/**
 * Linear Mapping layer to map between 3rd order tensors and dense matrices.
 */
template <
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::mat
>
using LinearMappingLayer = LinearLayer<InputDataType, OutputDataType>;

//! Layer traits for the linear layer.
template<
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<LinearLayer<InputDataType, OutputDataType> >
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
