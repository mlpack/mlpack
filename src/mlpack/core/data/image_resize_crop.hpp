/**
 * @file core/data/image_resize_crop.hpp
 * @author Omar Shrit
 *
 * Image resize and crop functionalities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_RESIZE_CROP_HPP
#define MLPACK_CORE_DATA_IMAGE_RESIZE_CROP_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/stb/stb.hpp>

#include "image_options.hpp"

namespace mlpack {
namespace data {

/**
 * Image resize/crop interfaces.
 */

/**
 * Resize one single image matrix or a set of images.
 *
 * This function should be used if the image is loaded as an armadillo matrix
 * and the number of cols equal to the Width and the number of rows equal
 * the Height of the image, or the total number of image pixels is equal to the
 * number of element in an armadillo matrix.
 *
 * The same applies if a set of images is loaded, but all of them need to have
 * identical dimension when loaded to this matrix.
 *
 * @param image The input matrix that contains the image to be resized.
 * @param info Contains relevant input images information.
 * @param newWidth The new requested width for the resized image.
 * @param newHeight The new requested height for the resized image.
 */
template<typename eT>
void ResizeImages(arma::Mat<eT>& images, ImageOptions& opts,
    const size_t newWidth, const size_t newHeight,
    const typename std::enable_if_t<std::is_floating_point<eT>::value>* = 0)
{
  // First check if we are resizing one image or a group of images, the check
  // is going to be different depending on the dimension.
  // If the user would like to resize a set of images of different dimensions,
  // then they need to consider passing them image by image. Otherwise, as
  // assume that all images have identical dimension and need to be resized.
  if (images.n_cols == 1)
  {
    if (images.n_elem != (opts.Width() * opts.Height() * opts.Channels()))
    {
      Log::Fatal << "ResizeImages(): dimensions mismatch: the number of pixels "
          << "is not equal to the dimension provided by the given ImageInfo."
          << std::endl;
    }
  }
  else
  {
    if (images.n_rows != (opts.Width() * opts.Height() * opts.Channels()))
    {
      Log::Fatal << "ResizeImages(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this "
          << "function iteratively." << std::endl;
    }
  }

  stbir_pixel_layout channels;
  if (opts.Channels() == 1)
  {
    channels = STBIR_1CHANNEL;
  }
  else if (opts.Channels() == 3)
  {
    channels = STBIR_RGB;
  }
  else
  {
    Log::Fatal << "ResizeImages(): number of channels should be either 1 or 3,"
        << " not " << opts.Channels() << "." << std::endl;
  }

  size_t newDimension = newWidth * newHeight * opts.Channels();
  arma::Mat<float> originalImagesFloat;
  arma::Mat<float> resizedFloatImages;

  // We need to covert to a float if we are resizing a double.
  if constexpr (std::is_same_v<eT, float>)
  {
    MakeAlias(originalImagesFloat, images, images.n_rows, images.n_cols);
  }
  else
  {
    originalImagesFloat =
        arma::conv_to<arma::Mat<float>>::from(std::move(images));
  }

  // recover the original min/max values for clamping.
  float minOriginal = images.min();
  float maxOriginal = images.max();

  resizedFloatImages.set_size(newDimension, images.n_cols);
  for (size_t i = 0; i < images.n_cols; ++i)
  {
    stbir_resize_float_linear(originalImagesFloat.colptr(i), opts.Width(),
          opts.Height(), 0, resizedFloatImages.colptr(i), newWidth, newHeight,
          0, channels);
  }

  images = arma::conv_to<arma::Mat<eT>>::from(std::move(resizedFloatImages));

  images.clamp(minOriginal, maxOriginal);

  opts.Width() = newWidth;
  opts.Height() = newHeight;
}

template<typename eT>
void ResizeImages(arma::Mat<eT>& images, ImageOptions& opts,
    const size_t newWidth, const size_t newHeight,
    const typename std::enable_if_t<std::is_integral_v<eT>>* = 0)
{
  // First check if we are resizing one image or a group of images, the check
  // is going to be different depending on the dimension.
  // If the user would like to resize a set of images of different dimensions,
  // then they need to consider passing them image by image. Otherwise, as
  // assume that all images have identical dimension and need to be resized.
  if (images.n_cols == 1)
  {
    if (images.n_elem != (opts.Width() * opts.Height() * opts.Channels()))
    {
      Log::Fatal << "ResizeImages(): dimensions mismatch: the number of pixels "
          << "is not equal to the dimension provided by the given ImageInfo."
          << std::endl;
    }
  }
  else
  {
    if (images.n_rows != (opts.Width() * opts.Height() * opts.Channels()))
    {
      Log::Fatal << "ResizeImages(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this "
          << "function iteratively." << std::endl;
    }
  }

  stbir_pixel_layout channels;
  if (opts.Channels() == 1)
  {
    channels = STBIR_1CHANNEL;
  }
  else if (opts.Channels() == 3)
  {
    channels = STBIR_RGB;
  }
  else
  {
    Log::Fatal << "ResizeImages(): number of channels should be either 1 or 3,"
        << " not " << opts.Channels() << "." << std::endl;
  }

  // This is required since STB only accept unsigned chars.
  // set the new matrix size for copy
  size_t newDimension = newWidth * newHeight * opts.Channels();
  arma::Mat<unsigned char> resizedImages;
  arma::Mat<unsigned char> originalImages;
  if constexpr (std::is_same_v<eT, unsigned char>)
  {
    MakeAlias(originalImages, images, images.n_rows, images.n_cols);
  }
  else
  {
    originalImages =
        arma::conv_to<arma::Mat<unsigned char>>::from(std::move(images));
  }

  resizedImages.set_size(newDimension, images.n_cols);

  for (size_t i = 0; i < images.n_cols; ++i)
  {
    stbir_resize_uint8_linear(originalImages.colptr(i), opts.Width(),
        opts.Height(), 0, resizedImages.colptr(i), newWidth, newHeight, 0,
        channels);
  }

  images = arma::conv_to<arma::Mat<eT>>::from(std::move(resizedImages));
  opts.Width() = newWidth;
  opts.Height() = newHeight;
}

/**
 * Resize & Crop one single image matrix or a set of images.
 *
 * This function should be used if the image is loaded as an armadillo matrix
 * and the number of cols equal to the Width and the number of rows equal
 * the Height of the image, or the total number of image pixels is equal to the
 * number of element in an armadillo matrix.
 *
 * The same applies if a set of images is loaded, but all of them need to have
 * identical dimension when loaded to this matrix.
 *
 * @param image The input matrix that contains the image to be resized.
 * @param info Contains relevant input images information.
 * @param newWidth The new requested width for the resized image.
 * @param newHeight The new requested height for the resized image.
 */
template<typename eT>
void ResizeCropImages(arma::Mat<eT>& images, ImageOptions& opts,
    const size_t newWidth, const size_t newHeight)
{
  float ratioW = static_cast<float>(newWidth)  /
      static_cast<float>(opts.Width());
  float ratioH = static_cast<float>(newHeight) /
      static_cast<float>(opts.Height());

  float largestRatio = ratioW > ratioH ? ratioW : ratioH;
  int midWidth = static_cast<int>(largestRatio * opts.Width());
  int midHeight = static_cast<int>(largestRatio * opts.Height());

  // Edge cases, what if the width / height value is odd ? then increase the
  // resize value to the closest pair number.
  // We have to avoid touching the image, of the user ask for it.
  // Add a condition to prevent cropping the image if the user did not ask for
  // any modification. Because cropping depends on the aspect ratio.
  if (ratioH != 1 || ratioW != 1)
  {
    if (midHeight % 2 != 0)
      midHeight = midHeight + 1;
    if (midWidth % 2 != 0)
      midWidth = midWidth + 1;

    ResizeImages(images, opts, midWidth, midHeight);
    int nColsCrop = midWidth > midHeight ? (midWidth - midHeight) : 0;
    int nRowsCrop = midHeight > midWidth ? (midHeight - midWidth) : 0;

    //temporary matrix to hold the images while being resized.
    arma::Mat<eT> tmpImages(newHeight * newWidth * opts.Channels(),
        images.n_cols);
    if (nRowsCrop != 0)
    {
      int cropUpDownEqually = (nRowsCrop / 2) * opts.Channels() * midWidth;
      tmpImages = images.rows(cropUpDownEqually,
          images.n_rows - cropUpDownEqually - 1);
    }

    #pragma omp parallel for
    for (size_t u = 0; u < images.n_cols; ++u)
    {
      if (nColsCrop != 0)
      {
        // Saving some memory by avoiding copying the images.
        // R into Row 1.
        // G into Row 2.
        // B into Row 3.
        // Cols are the Width, no change
        // Slices are the Height of the image instead of rows.
        arma::Cube<eT> cube(images.colptr(u), opts.Channels(), midWidth,
            midHeight, false, false);
        tmpImages.col(u) = vectorise(cube.cols((nColsCrop / 2),
              (cube.n_cols  - (nColsCrop / 2) - 1)));
      }
    }
    if (nRowsCrop != 0 || nColsCrop != 0)
    {
      images = std::move(tmpImages);
    }
  }
  opts.Width() = newWidth;
  opts.Height() = newHeight;
}

} // namespace data
} // namespace mlpack

#endif

