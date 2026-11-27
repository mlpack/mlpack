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

#ifndef MLPACK_DISABLE_STB

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
  arma::Mat<uint8_t> images;
  unsigned char* imageBuf = nullptr;
  size_t i = 0;

  // The channel count requested by the caller (0 means "use whatever the file
  // has"). stb delivers this many channels per pixel when it is nonzero, while
  // tempChannels reports the file's original channel count, which is not the
  // size of the returned buffer. Sizing the matrix from tempChannels reads past
  // the buffer when the two differ (e.g. an RGBA file loaded as RGB).
  const size_t requestedChannels = opts.Channels();

  while (i < files.size())
  {
    imageBuf = stbi_load(files.at(i).c_str(), &tempWidth, &tempHeight,
        &tempChannels, requestedChannels);
    if (!imageBuf)
    {
      std::stringstream oss;
      oss << "Load(): failed to load image '" << files.at(i) << "': "
              << stbi_failure_reason();
      return HandleError(oss, opts);
    }

    // Channels actually present in the returned buffer.
    const size_t loadedChannels = (requestedChannels != 0) ?
        requestedChannels : (size_t) tempChannels;

    if (opts.Width() == 0 || opts.Height() == 0)
    {
      opts.Width() = tempWidth;
      opts.Height() = tempHeight;
      opts.Channels() = loadedChannels;
    }
    dimension = opts.Width() * opts.Height() * opts.Channels();
    images.set_size(dimension, files.size());

    if ((size_t) tempWidth != opts.Width() ||
        (size_t) tempHeight != opts.Height() ||
        loadedChannels != opts.Channels())
    {
      stbi_image_free(imageBuf);
      std::stringstream oss;
      oss << "Load(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this"
          << " function iteratively." << std::endl;
      return HandleError(oss, opts);
    }
    images.col(i) = arma::Mat<uint8_t>(imageBuf, dimension, 1,
        false, true);
    stbi_image_free(imageBuf);
    i++;
  }
  matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(images));
  return true;
}

#else // MLPACK_DISABLE_STB

template<typename eT>
bool LoadImage(const std::vector<std::string>& /* files */,
               arma::Mat<eT>& /* matrix */,
               ImageOptions& opts)
{
  std::stringstream oss;
  oss << "Load(): image support was disabled at compile time "
         "(MLPACK_DISABLE_STB); rebuild without it to load images.";
  return HandleError(oss, opts);
}

#endif // MLPACK_DISABLE_STB

} // namespace mlpack

#endif
