/**
 * @file methods/ann/layer/minibatch_discrimination.hpp
 * @author Saksham Bansal
 *
 * Definition of the MiniBatchDiscrimination layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_HPP
#define MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_HPP

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MiniBatchDiscrimination layer. MiniBatchDiscrimination
 * is a layer of the discriminator that allows the discriminator to look at
 * multiple data examples in combination and perform what is called as
 * mini-batch discrimination.
 * This helps prevent the collapse of the generator parameters to a setting
 * where it emits the same point. This happens because normally a
 * discriminator will process each example independently and there will be
 * no mechanism to diversify the outputs of the generator.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Goodfellow2016,
 *   author  = {Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung,
 *              Alec Radford, Xi Chen},
 *   title   = {Improved Techniques for Training GANs},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1606.03498},
 * }
 * @endcode
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
class MiniBatchDiscrimination
{
 public:
  //! Create the MiniBatchDiscrimination object.
  MiniBatchDiscrimination();

  /**
   * Create the MiniBatchDiscrimination layer object using the specified
   * number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param features The number of features to compute for each dimension.
   */
  MiniBatchDiscrimination(const size_t inSize,
                          const size_t outSize,
                          const size_t features);

  /**
   * Reset the layer parameter.
   */
  void Reset();

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed-backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the
   * feed-forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param * (error) The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& /* error */,
                arma::Mat<eT>& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

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
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored dimensions of weight.
  size_t A, B, C;

  //! Locally-stored input batch size.
  size_t batchSize;

  //! Locally-stored temporary features object.
  arma::mat tempM;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored weight parameters.
  OutputDataType weight;

  //! Locally-stored features of input.
  arma::cube M;

  //! Locally-stored delta for features object.
  arma::cube deltaM;

  //! Locally-stored L1 distances between features.
  arma::cube distances;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored temporary delta object.
  OutputDataType deltaTemp;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MiniBatchDiscrimination

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "minibatch_discrimination_impl.hpp"

#endif
