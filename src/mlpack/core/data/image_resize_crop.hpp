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

#ifndef MLPACK_DISABLE_STB

/**
 * Image resize/crop interfaces.
 */

/**
 * Resize one single image matrix.
 *
 * This function should be used if the image is loaded as an armadillo matrix
 * and the number of cols equal to the Width and the number of rows equal
 * the Height of the image, or the total number of image pixels is equal to the
 * number of element in an armadillo matrix.
 *
 * @param image The input matrix that contains the image to be resized.
 * @param info Contains relevant input images information.
 * @param newWidth The new requested width for the resized image.
 * @param newHeight The new requested height for the resized image.
 */
template<typename eT>
inline void Resize(arma::Mat<eT>& image, data::ImageInfo& info,
    const size_t newWidth, const size_t newHeight)
{
  if (image.n_elem != (info.Width() * info.Height() * info.Channels()))
  {
    std::ostringstream oss;
    oss << "Dimensions mismatch. Resize(): only applicable on one image."
      " Please check if the image is loaded correctly into the matrix."
      << std::endl;
    Log::Fatal << oss.str();
  }

  // This is required since STB only accept unsigned chars.
  // set the new matrix size for copy
  size_t newDimension = newWidth * newHeight * info.Channels();
  arma::Mat<unsigned char> tempDest(newDimension, 1);

  // Allocate buffer to STB
  unsigned char* buffer =
      (unsigned char*)malloc(newDimension * sizeof(unsigned char));

  arma::Mat<unsigned char> tempSrc =
      arma::conv_to<arma::Mat<unsigned char>>::from(image);

  stbir_resize_uint8(tempSrc.memptr(), info.Width(), info.Height(), 0,
                     buffer, newWidth, newHeight, 0,
                     info.Channels());
  memcpy(tempDest.memptr(), buffer, sizeof(unsigned char) * newDimension);
  image = arma::conv_to<arma::Mat<eT>>::from(tempDest);

  info.Width() = newWidth;
  info.Height() = newHeight;
  free(buffer);
}

/**
 * Resize the images matrix.
 *
 * @param images The input matrix that contains the images to be resized.
 * @param info Contains relevant input images information.
 * @param resizedImages The output matrix the contains the resized images.
 * @param newWidth The new requested width for the resized images.
 * @param newHeight The new requested height for the resized images.
 */
template<typename eT>
inline void ResizeImages(arma::Mat<eT>& images, data::ImageInfo& info,
    const size_t newWidth, const size_t newHeight)
{
  if (images.n_rows != (info.Width() * info.Height() * info.Channels()))
  {
    std::ostringstream oss;
    oss << "ResizeImage(): the resizedImage matrix need to have identical"
      " dimensions to the image matrix" << std::endl;
    Log::Fatal << oss.str();
  }

  size_t newDimension = newWidth * newHeight * info.Channels();
  arma::Mat<unsigned char> tempDest(newDimension, images.n_cols);

  for (size_t i = 0; i < images.n_cols; ++i)
  {
    // Allocate buffer to STB
    unsigned char* buffer =
        (unsigned char*)malloc(newDimension * sizeof(unsigned char));

    arma::Mat<unsigned char> tempSrc =
        arma::conv_to<arma::Mat<unsigned char>>::from(images.col(i));

    stbir_resize_uint8(tempSrc.memptr(), info.Width(), info.Height(), 0,
                       buffer, newWidth, newHeight, 0,
                       info.Channels());

    memcpy(tempDest.memptr(), buffer, sizeof(unsigned char) * newDimension);
    images.col(i) = arma::conv_to<arma::Mat<eT>>::from(tempDest);
    free(buffer);
  }
  info.Width() = newWidth;
  info.Height() = newHeight;
}

#else

/**
 * The following are a set of dummy empty functions that do not do anything,
 * but only provide API compatibility in the case of NOT compiling mlpack with
 * STB.
 *
 * STB is by default part of mlpack, and these functions can only be executed
 * if the user specify MLPACK_DISABLE_STB.
 *
 * @param image The input matrix that contains the image to be resized.
 * @param info Contains relevant input images information.
 * @param newWidth The new requested width for the resized image.
 * @param newHeight The new requested height for the resized image.
 */
template<typename eT>
inline void Resize(arma::Mat<eT>& image, data::ImageInfo& info,
    const size_t newWidth, const size_t newHeight)
{
  Log::Fatal << "Resize(): mlpack was not compiled with STB support, so images"
      << " cannot be Resized!" << std::endl;
}

template<typename eT>
inline void ResizeImages(arma::Mat<eT>& images, data::ImageInfo& info,
    const size_t newWidth, const size_t newHeight)
{
  Log::Fatal << "ResizeImages(): mlpack was not compiled with STB support,"
      << " so images cannot be Resized!" << std::endl;
}

#endif

} // namespace data
} // namespace mlpack

#endif

