/**
 * @file spectral_norm.hpp
 * @author Saksham Bansal
 *
 * Definition of the SpectralNorm layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SPECTRAL_NORM_HPP
#define MLPACK_METHODS_ANN_LAYER_SPECTRAL_NORM_HPP

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
 * Declaration of the SpectralNorm layer class. The layer can stabalize the
 * training of discriminator networks. Spectral normalization controls the
 * the Lipschitz constant of the discriminator function f by literally
 * constraining the spectral norm of each layer.
 *
 * This class will be a wrapper around existing layers. It will just modify the
 * calculation and updation of weights of the layer.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @inproceedings{Miyato2016SpectralNorm,
 *   title = {Spectral Normalization for Generative Adversarial Networks},
 *   author = {Takeru Miyato, Toshiki Kataoka, Masanori Koyama,
 *             Yuichi Yoshida},
 *   booktitle = {ICLR 2018},
 *   year = {2018}
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
class SpectralNorm
{
 public:
  //! Create the SpectralNorm object.
  SpectralNorm();

  /**
   * Create the SpectralNorm layer object.
   *
   * @param inSize The input size of the linear layer.
   * @param outSize The output size of the linear layer.
   * @param powerIterations The number of iterations used for
   *        power iteration method.
   */
  SpectralNorm(const size_t inSize,
               const size_t outSize,
               const size_t powerIterations = 1);

  //! Destructor to release allocated memory.
  ~SpectralNorm();

  /**
   * Reset the layer parameters.
   */
  void Reset();

  /**
   * Forward pass of the SpectralNorm layer. Calculates the weights of the
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

  //! Modify the wrapped layer.
  Linear<>*& Layer() { return wrappedLayer; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of iterations used for power iteration method.
  size_t powerIterations;

  //! Locally-stored delete visitor module object.
  DeleteVisitor deleteVisitor;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored delta visitor module object.
  DeltaVisitor deltaVisitor;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored wrapped layer.
  Linear<>* wrappedLayer;

  //! Locally stored number of elements in the weights of wrapped layer.
  size_t layerWeightSize;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored output parameter visitor module object.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;

  //! Locally-stored scalar parameter.
  OutputDataType scalarParameter;

  //! Locally-stored parameter vector.
  OutputDataType vectorParameter;

  //! Locally-stored parameters.
  OutputDataType weights;

  //! Locally-stored weight parameters for wrapped layer.
  OutputDataType weight;

  //! Locally-stored bias term parameters for wrapped layer.
  OutputDataType bias;

  //! Locally-stored first left and right singular vectors of the weight.
  arma::vec u, v;

  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;

  //! Locally-stored gradients of wrappedLayer.
  OutputDataType layerGradients;

  //! Locally-stored weights of wrappedLayer.
  OutputDataType layerWeights;

  //! Locally-stored spectral norm of the weight.
  double sigma;
}; // class SpectralNorm

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "spectral_norm_impl.hpp"

#endif
