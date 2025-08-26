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
               ImageOptions& opts)
{
  size_t dimension = 0;
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Load(): list of images is empty, please specify the files names.";
    handleError(oss, opts);
  }

  // @rcurtin I would recommend only testing the last element in the vector.
  // Even if testing is fast, if you have 2M images it will take some time to
  // examin all of them for no reason.
  // Especially that in our case all files needs to have the same dimension
  // when loading them, if not we have to do resize for all of them. So it only
  // make sense that all have the same format.
  // Also we will need to check what is the extension if the user has provided
  // ImageType by default in case of backward compatibility or if they did not
  // specify the image
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
    handleError(oss, opts);
  }
  std::cout << "we passed file detection stage" << std::endl;
  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;
  // I think we can get rid of these two if we will be able to assign directly
  // to an armadillo matrix, but I doubt this is possible
  unsigned char* imageBuf;
  float* floatImageBuf;
  size_t i = 0;
  // Check dimension if provided or not
  // if not, then call the function on the first image to get the dimension
  // Note that, similar to the Resize function, we need to document that the
  // user cannot load images with different dimension in the same matrix.
  if (opts.Width() == 0 || opts.Height() == 0)
  {
    if constexpr (std::is_same_v<eT, unsigned char> ||
                  std::is_same_v<eT, size_t>)
    {
      std::cout << "load an unsigned char" << std::endl;
      imageBuf = stbi_load(files.front().c_str(), &tempWidth, &tempHeight,
          &tempChannels, STBI_rgb);
      if (!imageBuf)
      {
        std::stringstream oss;
        oss << "Load(): failed to load image '" << files.front() << "': "
            << stbi_failure_reason() << ".";

        handleError(oss, opts);
      }
    }
    else if constexpr (std::is_same_v<eT, float> || std::is_same_v<eT, double>)
    {
      std::cout << "load a a double" << std::endl;
      floatImageBuf = stbi_loadf(files.at(i).c_str(), &tempWidth, &tempHeight,
          &tempChannels, opts.Channels());
      if (!floatImageBuf)
      {
        std::stringstream oss;
        oss << "Load(): failed to load image '" << files.front() << "': "
                << stbi_failure_reason();

        handleError(oss, opts);
      }
    }
    opts.Width() = tempWidth;
    opts.Height() = tempHeight;
    opts.Channels() = tempChannels;
    i++;
  }

  dimension = opts.Width() * opts.Height() * opts.Channels();

  arma::Mat<unsigned char> images;
  arma::Mat<float> floatImages;
  std::cout << "Allocated matrices" << std::endl;
  if constexpr (std::is_same_v<eT, float> || std::is_same_v<eT, double>)
    floatImages.set_size(dimension, files.size());
  else
    images.set_size(dimension, files.size());

  std::cout << "Allocated matrices 2" << std::endl;
  while (i < files.size())
  { // Good question to @rcurtin, what if the user is passing, double, should
    // we handle it as a unsigned char and then we are converting by default to
    // double at the end of the function ? Or treat it as a float, but in this
    // case, we need to modify a couple of tests that we have
    if constexpr (std::is_same_v<eT, unsigned char> ||
                  std::is_same_v<eT, size_t>)
    {
      std::cout << "load an unsigned char" << std::endl;
      imageBuf = stbi_load(files.at(i).c_str(), &tempWidth, &tempHeight,
          &tempChannels, opts.Channels());
      if (!imageBuf)
      {
        std::stringstream oss;
        oss << "Load(): failed to load image '" << files.front() << "': "
                << stbi_failure_reason();
      
        handleError(oss, opts);
      }

      // We need to do this check after loading every image to be sure that
      // images provided by the user have an identical dimensions.
      if (tempWidth != opts.Width() || tempHeight != opts.Height()
          || tempChannels != opts.Channels())
      {
        std::stringstream oss;
        oss << "Load(): dimension mismatch: in the case of "
            << "several images, please check that all the images have the same "
            << "dimensions; if not, load each image in one column and call this"
            << " function iteratively." << std::endl;
        handleError(oss, opts);
      }
      images.col(i) = arma::Mat<unsigned char>(imageBuf, dimension, 1,
          false, true);
      stbi_image_free(imageBuf);
    }
    else if constexpr (std::is_same_v<eT, float> || std::is_same_v<eT, double>)
    {
      std::cout << "load a a double" << std::endl;
      floatImageBuf = stbi_loadf(files.at(i).c_str(), &tempWidth, &tempHeight,
          &tempChannels, opts.Channels());
      if (!floatImageBuf)
      {
        std::stringstream oss;
        oss << "Load(): failed to load image '" << files.front() << "': "
                << stbi_failure_reason();

        handleError(oss, opts);
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
        handleError(oss, opts);
      }
      floatImages.col(i) = arma::Mat<float>(floatImageBuf, dimension, 1,
          false, true);
      stbi_image_free(floatImageBuf);
    }
    i++;
  }
  if constexpr (std::is_same<eT, float>::value || std::is_same_v<eT, double>)
  {
    std::cout << "conversion back to the original matrix should be happening" << std::endl;
    matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(floatImages));
  }
  else
  {
    matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(images));
  }

  std::cout << "passed loop" << std::endl;
  return true;
}

} // namespace data
} // namespace mlpack

#endif
