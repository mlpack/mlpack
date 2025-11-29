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
void LetterboxImages(arma::Mat<eT>& src,
                    ImageOptions& srcOpt,
                    const size_t width,
                    const size_t height,
                    const eT fillValue)
{
  const size_t expectedRows =
    srcOpt.Width() * srcOpt.Height() * srcOpt.Channels();

  if (expectedRows == 0)
  {
    std::ostringstream errMessage;
    errMessage << "LetterboxImages(): Dimensions cannot contain a zero."
      " Received: " << srcOpt.Width() << " x " << srcOpt.Height()
      << " x " << srcOpt.Channels() << ".";
    throw std::logic_error(errMessage.str());
  }

  if (src.n_rows == 0)
    throw std::logic_error("LetterboxImages(): Matrix rows cannot be zero.");

  if (src.n_rows != expectedRows)
  {
    std::ostringstream errMessage;
    errMessage << "LetterboxImages(): Expected size of image was "
      << expectedRows << " but received " << src.n_rows;
    throw std::logic_error(errMessage.str());
  }

  if (srcOpt.Channels() != 1 && srcOpt.Channels() != 3)
  {
    std::ostringstream errMessage;
    errMessage << "LetterboxImages(): Must have only 1 or 3 channels, but "
      "received " << srcOpt.Channels();
    throw std::logic_error(errMessage.str());
  }

  size_t newWidth, newHeight;
  if (width * 1. / srcOpt.Width() > height * 1. / srcOpt.Height())
  {
    newHeight = height;
    newWidth = srcOpt.Width() * height / srcOpt.Height();
  }
  else
  {
    newWidth = width;
    newHeight = srcOpt.Height() * width / srcOpt.Width();
  }

  const size_t numImages = src.n_cols;
  arma::Mat<eT> dest(width * height * srcOpt.Channels(), numImages,
                     arma::fill::none);

  // Resize, then embed src within dest.
  ResizeImages(src, srcOpt, newWidth, newHeight);
  arma::Cube<eT> cubeSrc, cubeDest;

  // Channels as rows, because assumption is that channels are
  // interleaved (see image_layout.hpp for more info).
  MakeAlias(cubeSrc, src, srcOpt.Channels() * srcOpt.Width(), srcOpt.Height(),
            numImages);
  MakeAlias(cubeDest, dest, srcOpt.Channels() * width, height, numImages);

  const size_t dx = (width - newWidth) / 2;
  const size_t dy = (height - newHeight) / 2;

  cubeDest.fill(fillValue);
  // Fill RGB
  cubeDest.subcube(dx * srcOpt.Channels(),
                   dy,
                   0,
                   ((srcOpt.Width() + dx) * srcOpt.Channels()) - 1,
                   srcOpt.Height() + dy - 1,
                   numImages - 1) = cubeSrc;

  src = std::move(dest);
  srcOpt = ImageOptions(width, height, srcOpt.Channels());
}

} // namespace data
} // namespace mlpack

#endif
