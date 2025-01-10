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

namespace mlpack {
namespace data {

#ifndef MLPACK_DISABLE_STB

/**
 * Image resize/crop interfaces.
 */

/**
 * Resize one single image matrix.
 *
 * @param image The input matrix that contains the image to be resized.
 * @param info Contains relevant input images information.
 * @param resizedImage The output matrix the contains the resized image.
 * @param newWidth The new requested width for the resized image.
 * @param newHeight The new requested height for the resized image.
 */
template<typename eT>
inline void ResizeImage(arma::Mat<eT>& image, data::ImageInfo& info,
    arma::Mat<eT> resizedImage, const size_t newWidth, const size_t newHeight)
{
  if (image.n_cols > 1 or resizedImage.n_cols > 1)
  {
    std::ostringstream oss;
    oss << "ResizeImage(): only applicable on one image. For several images"
      " please use ResizeImages()" << std::endl;
    Log::Fatal << oss.str();
  }

  arma::Mat<unsigned char> tempSrc(size(image), arma::fill::zeros);
  arma::Mat<unsigned char> tempDest(size(resizedImage), arma::fill::zeros);

  stbir_resize_uint8(tempSrc.colptr(0), info.Width(), info.Height(), 0,
                     tempDest.colptr(0), newWidth, newHeight, 0,
                     info.Channels());
  resizedImage = arma::conv_to<eT>::from(tempDest);
  info.Width() = newWidth;
  info.Height() = newHeight;
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
    arma::Mat<eT> resizedImages, const size_t newWidth, const size_t newHeight)
{
  if (images.n_cols != resizedImages.n_cols)
  {
    std::ostringstream oss;
    oss << "ResizeImage(): the resizedImage matrix need to have identical"
      " dimensions to the image matrix" << std::endl;
    Log::Fatal << oss.str();
  }

  arma::Mat<unsigned char> tempSrc(size(images), arma::fill::zeros);
  arma::Mat<unsigned char> tempDest(size(resizedImages), arma::fill::zeros);

  for (size_t i = 0; i < images.n_cols; ++i)
  {
    stbir_resize_uint8(tempSrc.colptr(i), info.Width(), info.Height(), 0,
                       tempDest.colptr(i), newWidth, newHeight, 0,
                       info.Channels());
  }
  resizedImages = arma::conv_to<eT>::from(tempDest);
  info.Width() = newWidth;
  info.Height() = newHeight;
}

#endif

} // namespace data
} // namespace mlpack

#endif

