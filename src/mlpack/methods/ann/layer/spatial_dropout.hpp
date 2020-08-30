/**
 * @file methods/ann/layer/spatial_dropout.hpp
 * @author Anjishnu Mukherjee
 *
 * Definition of the SpatialDropout class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SPATIAL_DROPOUT_HPP
#define MLPACK_METHODS_ANN_LAYER_SPATIAL_DROPOUT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the SpatialDropout layer.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Tompson15,
 *   author    = {Jonathan Tompson, Ross Goroshin, Arjun Jain,
 *                Yann LeCun, Christopher Bregler},
 *   title     = {Efficient Object Localization Using Convolutional Networks},
 *   journal   = {CoRR},
 *   volume    = {abs/1411.4280},
 *   year      = {2015},
 *   url       = {https://arxiv.org/abs/1411.4280},
 *   eprint    = {1411.4280},
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
class SpatialDropout
{
 public:
  //! Create the SpatialDropout object.
  SpatialDropout();
  /**
   * Create the SpatialDropout object using the specified parameters.
   *
   * @param size The number of channels of each input image.
   * @param ratio The probability of each channel getting dropped.
   */
  SpatialDropout(const size_t size, const double ratio = 0.5);

  /**
   * Ordinary feed forward pass of the SpatialDropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of the SpatialDropout layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the number of channels.
  size_t Size() const { return size; }

  //! Modify the number of channels.
  size_t& Size() { return size; }

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the probability value.
  double Ratio() const { return ratio; }

  //! Modify the probability value.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored mast object.
  OutputDataType mask;

  //! The number of channels of each input image.
  size_t size;

  //! The probability of each channel getting dropped.
  double ratio;

  //! The scale fraction.
  double scale;

  //! A boolean used to do some internal calculations once initially.
  bool reset;

  //! The number of images in the batch.
  size_t batchSize;

  //! The number of pixels in each feature map.
  size_t inputSize;

  //! If true dropout and scaling are disabled.
  bool deterministic;
}; // class SpatialDropout

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "spatial_dropout_impl.hpp"

#endif
