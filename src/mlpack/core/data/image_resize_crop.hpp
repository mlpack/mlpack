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

#include "image_info.hpp"

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
inline void ResizeImages(arma::Mat<eT>& images, data::ImageInfo& info,
    const size_t newWidth, const size_t newHeight)
{
  // First check if we are resizing one image or a group of images, the check
  // is going to be different depending on the dimension.
  // If the user would like to resize a set of images of different dimensions,
  // then they need to consider passing them image by image. Otherwise, as
  // assume that all images have identical dimension and need to be resized.
  if (images.n_cols == 1)
  {
    if (images.n_elem != (info.Width() * info.Height() * info.Channels()))
    {
      Log::Fatal << "ResizeImages(): dimensions mismatch: the number of pixels "
          << "is not equal to the dimension provided by the given ImageInfo."
          << std::endl;
    }
  }
  else
  {
    if (images.n_rows != (info.Width() * info.Height() * info.Channels()))
    {
      Log::Fatal << "ResizeImages(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this "
          << "function iteratively." << std::endl;
    }
  }

  stbir_pixel_layout channels;
  if (info.Channels() == 1)
  {
    channels = STBIR_1CHANNEL;
  }
  else if (info.Channels() == 3)
  {
    channels = STBIR_RGB;
  }

  // This is required since STB only accept unsigned chars.
  // set the new matrix size for copy
  size_t newDimension = newWidth * newHeight * info.Channels();
  arma::Mat<float> resizedFloatImages;
  arma::Mat<unsigned char> resizedImages;

  // This is not optimal, but I do not want to allocate memory for nothing.
  if (std::is_same<eT, float>::value)
    resizedFloatImages.set_size(newDimension, images.n_cols);
  else
    resizedImages.set_size(newDimension, images.n_cols);

  for (size_t i = 0; i < images.n_cols; ++i)
  {
    if constexpr (std::is_same<eT, unsigned char>::value)
    {
      stbir_resize_uint8_linear(images.colptr(i), info.Width(), info.Height(),
          0, resizedImages.colptr(i), newWidth, newHeight, 0, channels);
    }
    else if constexpr (std::is_same<eT, float>::value)
    {
      stbir_resize_float_linear(images.colptr(i), info.Width(), info.Height(),
          0, resizedFloatImages.colptr(i), newWidth, newHeight, 0, channels);
    }
    else
    {
      arma::Mat<unsigned char> tempSrc =
          arma::conv_to<arma::Mat<unsigned char>>::from(images);

      stbir_resize_uint8_linear(tempSrc.colptr(i), info.Width(), info.Height(),
          0, resizedImages.colptr(i), newWidth, newHeight, 0, channels);
    }
  }

  if (std::is_same<eT, float>::value)
  {
    // The conv_to is needed here so that this code compiles even when this
    // branch isn't taken.
    images = arma::conv_to<arma::Mat<eT>>::from(std::move(resizedFloatImages));
  }
  else
  {
    images = arma::conv_to<arma::Mat<eT>>::from(std::move(resizedImages));
  }
  info.Width() = newWidth;
  info.Height() = newHeight;
}

} // namespace data
} // namespace mlpack

#endif

