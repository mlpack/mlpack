// Temporarily drop.
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

#include "layer.hpp"

namespace mlpack {

// TODO: this could likely use inputDimensions to remove the `size` parameter!
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
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class SpatialDropoutType : public Layer<InputType, OutputType>
{
 public:
  //! Create the SpatialDropout object.
  SpatialDropoutType();

  /**
   * Create the SpatialDropout object using the specified parameters.
   *
   * @param size The number of channels of each input image.
   * @param ratio The probability of each channel getting dropped.
   */
  SpatialDropoutType(const size_t size, const double ratio = 0.5);

  //! Clone the SpatialDropoutType object. This handles polymorphism correctly.
  SpatialDropoutType* Clone() const { return new SpatialDropoutType(*this); }

  /**
   * Ordinary feed forward pass of the SpatialDropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of the SpatialDropout layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input, const OutputType& gy, OutputType& g);

  //! Get the number of channels.
  size_t Size() const { return size; }

  //! Modify the number of channels.
  size_t& Size() { return size; }

  //! Get the probability value.
  double Ratio() const { return ratio; }

  //! Modify the probability value.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored mast object.
  OutputType mask;

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
}; // class SpatialDropout

// Convenience typedefs.

// Standard SpatialDropout layer.
using SpatialDropout = SpatialDropoutType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "spatial_dropout_impl.hpp"

#endif
