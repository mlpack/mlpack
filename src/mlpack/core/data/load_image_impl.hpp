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

// @rcurtin this function is a trial for now
bool mlpackException(const std::stringstream& oss, const DataOptions& opts)
{
  bool success;
  if (opts.Fatal())
  {
    Log::Fatal << oss.str() << std::endl;
    success = false;
  }
  else
  {
    Log::Warn << oss.str() << std::endl;
    success = true;
  }
  return success;
}

// Image loading API for multiple files.
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageOptions& opts)
{
  size_t dimension = 0;
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Load(): list of images is empty, please specify the files names."
    mlpackException(oss, opts); 
  }

  if (!opts.ImageFormatSupported(files.back()))
  {
    std::stringstream oss;
    oss << "Load(): file type " << Extension(files.back())
        << " isn't supported. Currently it supports: ";
    auto x = opts.LoadFileTypes();
    for (auto extension : x)
      oss << " " << extension;
    mlpackException(oss, opts);
  }

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;
  // I think we can get rid of these two if we will be able to assign directly
  // to an armadillo matrix, but I doubt this is possible
  unsigned char* imageBuf;
  float* floatImageBuf;
  // Check dimension if provided or not
  // if not, then call the function on the first image to get the dimension
  // Note that, similar to the Resize function, we need to document that the
  // user cannot load images with different dimension in the same matrix.
  if (opts.Width() == 0 || opts.Height() == 0)
  {
    imageBuf = stbi_load(files.front().c_str(), &tempWidth, &tempHeight,
        &tempChannels, STBI_rgb);
    if (!imageBuf)
    {
      std::stringstream oss;
      oss << "Load(): failed to load image '" << files.front() << "': "
              << stbi_failure_reason();
    
      mlpackException(oss, opts);
    }
    
    opts.Width() = tempWidth;
    opts.Height() = tempHeight;
    opts.Channels() = tempChannels;
  }

  dimension = opts.Width() * opts.Height() * opts.Channels();

  arma::Mat<unsigned char> images;
  arma::Mat<float> floatImages;

  if constexpr (std::is_same<eT, float>::value)
    floatImages.set_size(dimension, files.size());
  else
    images.set_size(dimension, files.size());

  for (size_t i = 0; i < files.size(); ++i)
  {
    if constexpr (std::is_same<eT, unsigned char>::value)
    {
      imageBuf = stbi_load(files.at(i).c_str(), &tempWidth, &tempHeight,
          &tempChannels, opts.Channels());
      if (!imageBuf)
      {
        std::stringstream oss;
        oss << "Load(): failed to load image '" << files.front() << "': "
                << stbi_failure_reason();
      
        mlpackException(oss, opts);
      }

      // We need to do this check after loading every image to be sure that
      // images provided by the user have an identical dimensions.
      if (tempWidth != opts.Width() && tempHeight != opts.Height())
      {
        std::stringstream oss;
        oss << "Load(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this "
          << "function iteratively." << std::endl;
        mlpackException(oss, opts);
      }
      images.colptr(i) = arma::Mat<unsigned char>(imageBuf, dimension, 1,
          true, true);
      stbi_image_free(imageBuf);
    }
    else if constexpr (std::is_same<eT, float>::value)
    {
      floatImageBuf = stbi_loadf(files.at(i).c_str(), &tempWidth, &tempHeight,
          &tempChannels, opts.Channels());
      if (!floatImageBuf)
      {
        std::stringstream oss;
        oss << "Load(): failed to load image '" << files.front() << "': "
                << stbi_failure_reason();
      
        mlpackException(oss, opts);
      }

      // We need to do this check after loading every image to be sure that
      // images provided by the user have an identical dimensions.
      if (tempWidth != opts.Width() && tempHeight != opts.Height())
      {
        std::stringstream oss;
        oss << "Load(): dimension mismatch: in the case of "
          << "several images, please check that all the images have the same "
          << "dimensions; if not, load each image in one column and call this "
          << "function iteratively." << std::endl;
        mlpackException(oss, opts);
      }
      floatImages.colptr(i) = arma::Mat<unsigned char>(imageBuf, dimension, 1,
          true, true);
      stbi_image_free(floatImageBuf);
    }
  }
  if constexpr (std::is_same<et, float>::value)
  {
    matrix = arma::conv_to<arma::Mat<eT>>::from(floatImages);
  }
  else
  {
    matrix = arma::conv_to<arma::Mat<eT>>::from(images);
  }

  return true;
}

} // namespace data
} // namespace mlpack

#endif
