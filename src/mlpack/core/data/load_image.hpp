/**
 * @file core/data/load_image.hpp
 * @author Mehul Kumar Nirala
 * @author Omar Shrit
 *
 * Implementation of image loading functionality via STB.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_LOAD_IMAGE_HPP
#define MLPACK_CORE_DATA_LOAD_IMAGE_HPP

#include <mlpack/core/stb/stb.hpp>

namespace mlpack {
namespace data {

template<typename eT>
bool LoadImage(const std::vector<std::string>& files,
               arma::Mat<eT>& matrix,
               ImageOptions& opts)
{
  size_t dimension = 0;
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Load(): list of images is empty, please specify the files names.";
    return HandleError(oss, opts);
  }

  if (opts.Format() == FileType::ImageType ||
      opts.Format() == FileType::AutoDetect)
  {
    DetectFromExtension<arma::Mat<eT>, ImageOptions>(files.back(), opts);
    if (!opts.loadType.count(Extension(files.back())))
    {
      std::stringstream oss;
      oss << "Load(): image type " << opts.FileTypeToString()
        << " not supported. Supported formats: ";
      for (const auto& x : opts.loadType)
        oss << " " << x;
      return HandleError(oss, opts);
    }
  }

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;
  arma::Mat<unsigned char> images;
  unsigned char* imageBuf = nullptr;
  size_t i = 0;

  while (i < files.size())
  {
    imageBuf = stbi_load(files.at(i).c_str(), &tempWidth, &tempHeight,
        &tempChannels, opts.Channels());
    if (!imageBuf)
    {
      std::stringstream oss;
      oss << "Load(): failed to load image '" << files.at(i) << "': "
              << stbi_failure_reason();
      return HandleError(oss, opts);
    }
    if (opts.Width() == 0 || opts.Height() == 0)
    {
      opts.Width() = tempWidth;
      opts.Height() = tempHeight;
      opts.Channels() = tempChannels;
    }
    dimension = opts.Width() * opts.Height() * opts.Channels();
    images.set_size(dimension, files.size());

    if ((size_t) tempWidth != opts.Width() ||
        (size_t) tempHeight != opts.Height() ||
        (size_t) tempChannels != opts.Channels())
    {
      std::stringstream oss;
      oss << "Load(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this"
          << " function iteratively." << std::endl;
      return HandleError(oss, opts);
    }
    images.col(i) = arma::Mat<unsigned char>(imageBuf, dimension, 1,
        false, true);
    stbi_image_free(imageBuf);
    i++;
  }
  matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(images));
  return true;
}

} // namespace data
} // namespace mlpack

#endif
