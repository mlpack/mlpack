/**
 * @file core/data/image_letterbox.hpp
 * @author Andrew Furey
 *
 * Apply letterbox transform to an image.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_LETTERBOX_HPP
#define MLPACK_CORE_DATA_IMAGE_LETTERBOX_HPP

#include <mlpack/prereqs.hpp>

#include "image_options.hpp"
#include "image_resize_crop.hpp"

namespace mlpack {
namespace data {

/**
 * Resize an image to `imgSize` x `imgSize` while keeping the original
 * image's aspect ratio. Fill in white space with `fillValue`.
 *
 * @param src The source matrix that contains the images.
 * @param srcOpt Contains relevant information on each source image.
 * @param dest The destination matrix that will have the source image
          embedded onto it
 * @param imgSize The width and height of each output image.
 * @param fillValue The whitespace value.
 */
template<typename eT>
void LetterboxImage(arma::Mat<eT>& src,
                    ImageOptions& srcOpt,
                    const size_t imgSize,
                    const eT fillValue)
{
  const size_t expectedRows =
    srcOpt.Width() * srcOpt.Height() * srcOpt.Channels();

  if (src.n_rows != expectedRows)
  {
    std::ostringstream errMessage;
    errMessage << "LetterboxImage(): Expected size of image was "
      << expectedRows << " but received " << src.n_rows;
    throw std::logic_error(errMessage.str());
  }

  if (src.n_cols != 1)
  {
    std::ostringstream errMessage;
    errMessage << "LetterboxImage(): Expected 1 image but received "
      << src.n_cols;
    throw std::logic_error(errMessage.str());
  }

  if (srcOpt.Channels() != 1 && srcOpt.Channels() != 3)
  {
    std::ostringstream errMessage;
    errMessage << "LetterboxImage(): Must have only 1 or 3 channels, but "
      "received " << srcOpt.Channels();
    throw std::logic_error(errMessage.str());
  }

  size_t width, height;
  if (srcOpt.Width() < srcOpt.Height())
  {
    height = imgSize;
    width = srcOpt.Width() * imgSize / srcOpt.Height();
  }
  else
  {
    width = imgSize;
    height = srcOpt.Height() * imgSize / srcOpt.Width();
  }

  arma::Mat<eT> dest(imgSize * imgSize * srcOpt.Channels(), 1,
                     arma::fill::none);

  // Resize, then embed src within dest.
  ResizeImages(src, srcOpt, width, height);
  arma::Cube<eT> cubeSrc, cubeDest;

  // Channels as rows, because default assumption is that channels are
  // interleaved (see image_layout.hpp for more info).
  MakeAlias(cubeSrc, src, srcOpt.Channels(), srcOpt.Width(), srcOpt.Height());
  MakeAlias(cubeDest, dest, srcOpt.Channels(), imgSize, imgSize);

  const size_t dx = (imgSize - width) / 2;
  const size_t dy = (imgSize - height) / 2;

  cubeDest.fill(fillValue);
  // Fill RGB
  cubeDest.subcube(0, dx, dy, srcOpt.Channels() - 1, srcOpt.Width() + dx - 1,
    srcOpt.Height() + dy - 1) = cubeSrc;

  src = std::move(dest);
  srcOpt = ImageOptions(imgSize, imgSize, srcOpt.Channels());
}

} // namespace data
} // namespace mlpack

#endif

