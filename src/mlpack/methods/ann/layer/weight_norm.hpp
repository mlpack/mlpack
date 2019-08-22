/**
 * @file weight_norm.hpp
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
#include "layer_types.hpp"

#include "../visitor/delete_visitor.hpp"
#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"
#include "../visitor/reset_visitor.hpp"
#include "../visitor/weight_size_visitor.hpp"
#include "../visitor/weight_set_visitor.hpp"

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
 *   year = {2016}
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam CustomLayers Additional custom layers that can be added.
 */
template <
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat,
  typename... CustomLayers
>
class WeightNorm
{
 public:
  /**
   * Create the WeightNorm layer object.
   *
   * @param layer The layer whose weights are needed to be normalized.
   */
  WeightNorm(LayerTypes<CustomLayers...> layer = LayerTypes<CustomLayers...>());

  //! Destructor to release allocated memory.
  ~WeightNorm();

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
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);

  /**
   * Backward pass through the layer. This function calls the Backward()
   * function of the wrapped layer.
   *
   * @param input The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  /**
   * Calculate the gradient using the output delta, input activations and the
   * weights of the wrapped layer.
   *
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& gradient);

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the wrapped layer.
  LayerTypes<CustomLayers...> const& Layer() { return wrappedLayer; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of bias elements in the weights of wrapped layer.
  size_t biasWeightSize;

  //! Locally-stored delete visitor module object.
  DeleteVisitor deleteVisitor;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored delta visitor module object.
  DeltaVisitor deltaVisitor;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored wrapped layer.
  LayerTypes<CustomLayers...> wrappedLayer;

  //! Locally stored number of elements in the weights of wrapped layer.
  size_t layerWeightSize;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored output parameter visitor module object.
  OutputParameterVisitor outputParameterVisitor;

  //! Reset the gradient for all modules that implement the Gradient function.
  void ResetGradients(arma::mat& gradient);

  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;

  //! Locally-stored scalar parameter.
  OutputDataType scalarParameter;

  //! Locally-stored parameter vector.
  OutputDataType vectorParameter;

  //! Locally-stored parameters.
  OutputDataType weights;

  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;

  //! Locally-stored gradients of wrappedLayer.
  OutputDataType layerGradients;

  //! Locally-stored weights of wrappedLayer.
  OutputDataType layerWeights;
}; // class WeightNorm

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "weight_norm_impl.hpp"

#endif
