/**
 * @file core/data/load_image_impl.hpp
 * @author Mehul Kumar Nirala
 * @author Omar Shrit
 * @author Ryan Curtin
 *
 * An image loading utility implementation via STB.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_IMAGE_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_IMAGE_IMPL_HPP

// In case it hasn't been included yet.
#include "load_image.hpp"

namespace mlpack {
namespace data {

template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  DataOptionsType tmpOpts(opts);
  return Load(files, matrix, tmpOpts);
}

// Image loading API for multiple files.
// To be organized in the next PR when deprecating the old API.
template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  bool success = false;
  if (files.empty())
  {
    return HandleError("Load(): given set of filenames is empty;"
        " loading failed.", opts);
  }

  DetectFromExtension<arma::Mat<eT>>(files.back(), opts);
  const bool isImageFormat = (opts.Format() == FileType::PNG ||
      opts.Format() == FileType::JPG || opts.Format() == FileType::PNM ||
      opts.Format() == FileType::BMP || opts.Format() == FileType::GIF ||
      opts.Format() == FileType::PSD || opts.Format() == FileType::TGA ||
      opts.Format() == FileType::PIC || opts.Format() == FileType::ImageType);

  if (isImageFormat)
  {
    ImageOptions imgOpts(std::move(opts));
    success = LoadImage(files, matrix, imgOpts);
    opts = std::move(imgOpts);
  }
  else
  {
    TextOptions txtOpts(std::move(opts));
    success = LoadNumericVector(files, matrix, txtOpts);
    opts = std::move(txtOpts);
  }
  return success;
}

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
