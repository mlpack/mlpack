/**
 * @file load_image_impl.hpp
 * @author Mehul Kumar Nirala
 *
 * An image loading utility implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_LOAD_IMAGE_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_IMAGE_IMPL_HPP

// In case it hasn't been included yet.
#include "load.hpp"

namespace mlpack {
namespace data {

#ifdef HAS_STB // Compile this only if stb is present.

// Image loading API.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool /* fatal */,
          const bool transpose)
{
  Timer::Start("loading_image");
  unsigned char* image;

  if (!ImageFormatSupported(filename))
  {
    std::ostringstream oss;
    oss << "File type " << Extension(filename) << " not supported.\n";
    oss << "Currently it supports ";
    for (auto extension : loadFileTypes)
      oss << " " << extension;
    oss << std::endl;
    throw std::runtime_error(oss.str());
    return false;
  }

  stbi_set_flip_vertically_on_load(transpose);

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;

  // For grayscale images.
  if (info.Channels() == 1)
  {
    image = stbi_load(filename.c_str(), &tempWidth, &tempHeight, &tempChannels,
        STBI_grey);
  }
  else
  {
    image = stbi_load(filename.c_str(), &tempWidth, &tempHeight, &tempChannels,
        STBI_rgb);
  }

  if (tempWidth <= 0 || tempHeight <= 0)
  {
    std::ostringstream oss;
    oss << "Image '" << filename << "' not found." << std::endl;
    free(image);
    throw std::runtime_error(oss.str());

    return false;
  }

  info.Width() = tempWidth;
  info.Height() = tempHeight;
  info.Channels() = tempChannels;

  // Copy image into armadillo Mat.
  matrix = arma::Mat<unsigned char>(image, info.Width() * info.Height() *
      info.Channels(), 1, true, true);

  // Free the image pointer.
  free(image);
  Timer::Stop("loading_image");
  return true;
}

// Image loading API for multiple files.
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  if (files.size() == 0)
  {
    std::ostringstream oss;
    oss << "Files vector is empty." << std::endl;

    throw std::runtime_error(oss.str());
    return false;
  }

  arma::Mat<unsigned char> img;
  bool status = Load(files[0], img, info, fatal, transpose);

  // Decide matrix dimension using the image height and width.
  matrix.set_size(info.Width() * info.Height() * info.Channels(), files.size());
  matrix.col(0) = img;

  for (size_t i = 1; i < files.size() ; i++)
  {
    arma::Mat<unsigned char> colImg(matrix.colptr(i), matrix.n_rows, 1,
        false, true);
    status &= Load(files[i], colImg, info, fatal, transpose);
  }
  return status;
}

#else // No STB.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false,
          const bool transpose = true)
{
  throw std::runtime_error("Load(): HAS_STB is not defined, "
      "so STB is not available and images cannot be loaded!");
}

template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false,
          const bool transpose = true)
{
  throw std::runtime_error("Load(): HAS_STB is not defined, "
      "so STB is not available and images cannot be loaded!");
}
#endif // HAS_STB.

} // namespace data
} // namespace mlpack

#endif
