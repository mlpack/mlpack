/**
 * @file core/data/save_image.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of save image functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_IMAGE_HPP
#define MLPACK_CORE_DATA_SAVE_IMAGE_HPP

#include <mlpack/core/stb/stb.hpp>
#include <mlpack/core/math/make_alias.hpp>

namespace mlpack {
namespace data {

template<typename eT>
bool SaveImage(const std::vector<std::string>& files,
               const arma::Mat<eT>& matrix,
               ImageOptions& opts)
{
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Save(): vector of image files is empty; nothing to save.";
    return HandleError(oss, opts);
  }

  // Check if we do have any type that is not supported.
  if (opts.Format() == FileType::ImageType ||
      opts.Format() == FileType::AutoDetect)
  {
    for (size_t i = 0; i < files.size() ; ++i)
    {
      if (!opts.saveType.count(Extension(files.at(i))))
      {
        std::stringstream oss;
        oss << "Save(): file type " << opts.FileTypeToString()
            << " isn't supported. Currently image saving supports: ";
        for (const auto& x : opts.saveType)
          oss << "  " << x;
        oss << "." << std::endl;
        return HandleError(oss, opts);
      }
    }
  }

  size_t dimension = opts.Width() * opts.Height() * opts.Channels() *
      files.size();
  // We only need to check the rows since it is a matrix.
  if (dimension != matrix.n_rows * matrix.n_cols)
  {
    std::stringstream oss;
    oss << "data::Save(): The given image dimensions, Width: " << opts.Width()
        << ", Height: " << opts.Height() << ", Channels: "<< opts.Channels()
        << " do not match the dimensions of the matrix to be saved!";
    return HandleError(oss, opts);
  }
  // Unfortunately we cannot move because matrix is const.
  arma::Mat<unsigned char> tempMatrix =
      arma::conv_to<arma::Mat<unsigned char>>::from(matrix);
  bool success = false;
  for (size_t i = 0; i < files.size() ; ++i)
  {
    // Update opts.Format() at each iteration.
    DetectFromExtension<arma::Mat<eT>, ImageOptions>(files.at(i), opts);
    if (opts.Format() == FileType::PNG)
    {
      success = stbi_write_png(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), tempMatrix.colptr(i),
          opts.Width() * opts.Channels());
    }
    else if (opts.Format() == FileType::BMP)
    {
      success = stbi_write_bmp(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), tempMatrix.colptr(i));
    }
    else if (opts.Format() == FileType::TGA)
    {
      success = stbi_write_tga(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), tempMatrix.colptr(i));
    }
    else if (opts.Format() == FileType::JPG)
    {
      success = stbi_write_jpg(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), tempMatrix.colptr(i), opts.Quality());
    }

    if (!success)
    {
      std::stringstream oss;
      oss << "Save(): error saving image to '" << files.at(i) << "'.";
      return HandleError(oss, opts);
    }
  }

  return success;
}

} //namespace data
} //namespace mlpack

#endif
