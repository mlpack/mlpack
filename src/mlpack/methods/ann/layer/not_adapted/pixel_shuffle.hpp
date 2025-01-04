/**
 * @file methods/ann/layer/pixel_shuffle.hpp
 * @author Anjishnu Mukherjee
 * @author Abhinav Anand
 *
 * Definition of the PixelShuffle class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_HPP
#define MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the PixelShuffle layer.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Shi16,
 *   author    = {Wenzhe Shi, Jose Caballero,Ferenc Huszár, Johannes Totz,
 *               Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang},
 *   title     = {Real-Time Single Image and Video Super-Resolution Using an
 *               Efficient Sub-Pixel Convolutional Neural Network},
 *   journal   = {CoRR},
 *   volume    = {abs/1609.05158},
 *   year      = {2016},
 *   url       = {https://arxiv.org/abs/1609.05158},
 *   eprint    = {1609.05158},
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
class PixelShuffle
{
 public:
  //! Create the PixelShuffle object.
  PixelShuffle();

  /**
   * Create the PixelShuffle object using the specified parameters.
   * The number of input channels should be an integral multiple of the square
   * of the upscale factor.
   *
   * @param upscaleFactor The scaling factor for Pixel Shuffle.
   * @param height The height of each input image.
   * @param width The width of each input image.
   * @param size The number of channels of each input image.
   */
  PixelShuffle(const size_t upscaleFactor,
               const size_t height,
               const size_t width,
               const size_t size);

  /**
   * Ordinary feed forward pass of the PixelShuffle layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of the PixelShuffle layer.
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

  //! Get the upscale factor.
  size_t UpscaleFactor() const { return upscaleFactor; }

  //! Modify the upscale factor.
  size_t& UpscaleFactor() { return upscaleFactor; }

  //! Get the input image height.
  size_t InputHeight() const { return height; }

  //! Modify the input image height.
  size_t& InputHeight() { return height; }

  //! Get the input image width.
  size_t InputWidth() const { return width; }

  //! Modify the input image width.
  size_t& InputWidth() { return width; }

  //! Get the number of input channels.
  size_t InputChannels() const { return size; }

  //! Modify the number of input channels.
  size_t& InputChannels() { return size; }

  //! Get the output image height.
  size_t OutputHeight() const { return outputHeight; }

  //! Get the output image width.
  size_t OutputWidth() const { return outputWidth; }

  //! Get the number of output channels.
  size_t OutputChannels() const { return sizeOut; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The scaling factor for Pixel Shuffle.
  size_t upscaleFactor;

  //! The height of each input image.
  size_t height;

  //! The width of each input image.
  size_t width;

  //! The number of channels of each input image.
  size_t size;

  //! The number of images in the batch.
  size_t batchSize;

  //! The height of each output image.
  size_t outputHeight;

  //! The width of each output image.
  size_t outputWidth;

  //! The number of channels of each output image.
  size_t sizeOut;

  //! A boolean used to do some internal calculations once initially.
  bool reset;
}; // class PixelShuffle

} // namespace mlpack

// Include implementation.
#include "pixel_shuffle_impl.hpp"

#endif
