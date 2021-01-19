/**
 * @file methods/ann/layer/weight_norm.hpp
 * @author Toshal Agrawal
 *
 * Definition of the WeightNorm layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_HPP
#define MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Declaration of the WeightNorm layer class. The layer reparameterizes
 * the weight vectors in a neural network, decoupling the length of those weight
 * vectors from their direction. This reparameterization does not introduce any
 * dependencies between the examples in a mini-batch.
 *
 * This class will be a wrapper around existing layers. It will just modify the
 * calculation and updation of weights of the layer.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @inproceedings{Salimans2016WeightNorm,
 *   title = {Weight Normalization: A Simple Reparameterization to Accelerate
 *            Training of Deep Neural Networks},
 *   author = {Tim Salimans, Diederik P. Kingma},
 *   booktitle = {Neural Information Processing Systems 2016},
 *   year = {2016},
 *   url  = {https://arxiv.org/abs/1602.07868},
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename InputType = arma::mat,
  typename OutputType = arma::mat
>
class WeightNormType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the WeightNorm layer object.
   *
   * @param layer The layer whose weights are needed to be normalized.
   */
  WeightNormType(Layer<InputType, OutputType>* layer);

  //! Destructor to release allocated memory.
  ~WeightNormType();

  /**
   * Reset the layer parameters.
   */
  void Reset();

  /**
   * Forward pass of the WeightNorm layer. Calculates the weights of the
   * wrapped layer from the parameter vector v and the scalar parameter g.
   * It then calulates the output of the wrapped layer from the calculated
   * weights.
   *
   * @param input Input data for the layer.
   * @param output Resulting output activations.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Backward pass through the layer. This function calls the Backward()
   * function of the wrapped layer.
   *
   * @param input The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta, input activations and the
   * weights of the wrapped layer.
   *
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  OutputType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the wrapped layer.
  Layer<InputType, OutputType>* const& WrappedLayer() { return wrappedLayer; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of bias elements in the weights of wrapped layer.
  size_t biasWeightSize;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored wrapped layer.
  Layer<InputType, OutputType>* wrappedLayer;

  //! Locally stored number of elements in the weights of wrapped layer.
  size_t layerWeightSize;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Reset the gradient for all modules that implement the Gradient function.
  void ResetGradients(OutputType& gradient);

  //! Locally-stored scalar parameter.
  OutputType scalarParameter;

  //! Locally-stored parameter vector.
  OutputType vectorParameter;

  //! Locally-stored parameters.
  OutputType weights;

  //! Locally-stored gradients of wrappedLayer.
  OutputType layerGradients;

  //! Locally-stored weights of wrappedLayer.
  OutputType layerWeights;
}; // class WeightNormType.

// Standard WeightNorm layer.
typedef WeightNormType<arma::mat, arma::mat> WeightNorm;

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "weight_norm_impl.hpp"

#endif
