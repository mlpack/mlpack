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

// Image loading API for multiple files.
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageOptions& opts)
{
  std::cout << "this should not execute Load for Image" << std::endl;
  return LoadImage(files, matrix, opts);
}

template<typename eT>
bool LoadImage(const std::vector<std::string>& files,
               arma::Mat<eT>& matrix,
               ImageOptions& opts,
               const typename std::enable_if_t<
                  std::is_floating_point<eT>::value>* = 0)
{
  size_t dimension = 0;
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Load(): list of images is empty, please specify the files names.";
    HandleError(oss, opts);
  }

  if (opts.Format() == FileType::ImageType)
  {
    DetectFromExtension<arma::Mat<eT>, ImageOptions>(files.back(), opts);
  }
  if (!opts.loadType.count(Extension(files.back())))
  {
    std::stringstream oss;
    oss << "Load(): image type " << opts.FileTypeToString()
      << " not supported. Supported formats: ";
    for (const auto& x : opts.loadType)
      oss << " " << x;
    HandleError(oss, opts);
  }

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;
  float* imageBuf;
  arma::Mat<float> images;
  size_t i = 0;
  while (i < files.size())
  {
    imageBuf = stbi_loadf(files.at(i).c_str(), &tempWidth, &tempHeight,
        &tempChannels, opts.Channels());
    if (!imageBuf)
    {
      std::stringstream oss;
      oss << "Load(): failed to load image '" << files.front() << "': "
              << stbi_failure_reason();
      HandleError(oss, opts);
    }
    if (opts.Width() == 0 || opts.Height() == 0)
    {
      opts.Width() = tempWidth;
      opts.Height() = tempHeight;
      opts.Channels() = tempChannels;
    }
    dimension = opts.Width() * opts.Height() * opts.Channels();
    images.set_size(dimension, files.size());

    if (tempWidth != opts.Width() || tempHeight != opts.Height()
        || tempChannels != opts.Channels())
    {
      std::stringstream oss;
      oss << "Load(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this"
          << " function iteratively." << std::endl;
      HandleError(oss, opts);
    }
    images.col(i) = arma::Mat<float>(imageBuf, dimension, 1, false, true);
    stbi_image_free(imageBuf);
    i++;
  }
  matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(images));
  return true;
}

template<typename eT>
bool LoadImage(const std::vector<std::string>& files,
              arma::Mat<eT>& matrix,
              ImageOptions& opts,
              const typename std::enable_if_t<std::is_integral_v<eT>>* = 0)
{
  size_t dimension = 0;
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Load(): list of images is empty, please specify the files names.";
    HandleError(oss, opts);
  }

  if (opts.Format() == FileType::ImageType)
  {
    DetectFromExtension<arma::Mat<eT>, ImageOptions>(files.back(), opts);
  }
  if (!opts.loadType.count(Extension(files.back())))
  {
    std::stringstream oss;
    oss << "Load(): image type " << opts.FileTypeToString()
      << " not supported. Supported formats: ";
    for (const auto& x : opts.loadType)
      oss << " " << x;
    HandleError(oss, opts);
  }

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;
  arma::Mat<unsigned char> images;
  unsigned char* imageBuf;
  size_t i = 0;

  while (i < files.size())
  {
    std::cout << "load an unsigned char" << std::endl;
    imageBuf = stbi_load(files.at(i).c_str(), &tempWidth, &tempHeight,
        &tempChannels, opts.Channels());
    if (!imageBuf)
    {
      std::stringstream oss;
      oss << "Load(): failed to load image '" << files.front() << "': "
              << stbi_failure_reason();
      HandleError(oss, opts);
    }
    if (opts.Width() == 0 || opts.Height() == 0)
    {
      opts.Width() = tempWidth;
      opts.Height() = tempHeight;
      opts.Channels() = tempChannels;
    }
    dimension = opts.Width() * opts.Height() * opts.Channels();
    images.set_size(dimension, files.size());

    if (tempWidth != opts.Width() || tempHeight != opts.Height()
        || tempChannels != opts.Channels())
    {
      std::stringstream oss;
      oss << "Load(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this"
          << " function iteratively." << std::endl;
      HandleError(oss, opts);
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
