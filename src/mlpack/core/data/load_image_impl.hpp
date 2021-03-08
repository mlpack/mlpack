/**
 * @file core/data/load_image_impl.hpp
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

// Image loading API.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal)
{
  Timer::Start("loading_image");

  // STB loads into unsigned char matrices, so we may have to convert once
  // loaded.
  arma::Mat<unsigned char> tempMatrix;
  const bool result = LoadImage(filename, tempMatrix, info, fatal);

  // If fatal is true, then the program will have already thrown an exception.
  if (!result)
  {
    Timer::Stop("loading_image");
    return false;
  }

  matrix = arma::conv_to<arma::Mat<eT>>::from(tempMatrix);
  Timer::Stop("loading_image");
  return true;
}

// Image loading API for multiple files.
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal)
{
  if (files.size() == 0)
  {
    std::ostringstream oss;
    oss << "Load(): vector of image files is empty." << std::endl;

    if (fatal)
      Log::Fatal << oss.str();
    else
      Log::Warn << oss.str();

    return false;
  }

  arma::Mat<unsigned char> img;
  bool status = LoadImage(files[0], img, info, fatal);

  if (!status)
    return false;

  // Decide matrix dimension using the image height and width.
  arma::Mat<unsigned char> tmpMatrix(
      info.Width() * info.Height() * info.Channels(), files.size());
  tmpMatrix.col(0) = img;

  for (size_t i = 1; i < files.size() ; ++i)
  {
    arma::Mat<unsigned char> colImg(tmpMatrix.colptr(i), tmpMatrix.n_rows, 1,
        false, true);
    status = LoadImage(files[i], colImg, info, fatal);

    if (!status)
      return false;
  }

  matrix = arma::conv_to<arma::Mat<eT>>::from(tmpMatrix);
  return true;
}

} // namespace data
} // namespace mlpack

#endif
