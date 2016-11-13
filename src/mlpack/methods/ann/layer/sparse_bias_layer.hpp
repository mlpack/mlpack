/**
 * @file sparse_bias_layer.hpp
 * @author Tham Ngap Wei
 *
 * Definition of the SparseBiasLayer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SPARSE_BIAS_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_SPARSE_BIAS_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a bias layer design for sparse autoencoder.
 * The BiasLayer class represents a single layer of a neural network.
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
class SparseBiasLayer
{
 public:
  /**
   * Create the SparseBiasLayer object using the specified number of units and
   * bias parameter.
   *
   * @param outSize The number of output units.
   * @param batchSize The batch size used to train the network.
   * @param bias The bias value.
   */
  SparseBiasLayer(const size_t outSize, const size_t batchSize) :
      outSize(outSize),
      batchSize(batchSize)
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
    output = input + arma::repmat(weights, 1, input.n_cols);
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
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT>
  void Gradient(const InputType& /* input */,
                const arma::Mat<eT>& d,
                InputDataType& g)
  {
    g = arma::sum(d, 1) / static_cast<typename InputDataType::value_type>(
        batchSize);
  }

  //! Get the batch size
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size
  size_t& BatchSize() { return batchSize; }

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
    ar & data::CreateNVP(batchSize, "batchSize");
  }

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! The batch size used to train the network.
  size_t batchSize;

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
}; // class SparseBiasLayer

//! Layer traits for the bias layer.
template<typename InputDataType, typename OutputDataType
>
class LayerTraits<SparseBiasLayer<InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = true;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

} // namespace ann
} // namespace mlpack

#endif
