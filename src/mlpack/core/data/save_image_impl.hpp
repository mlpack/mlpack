/**
 * @file core/data/save_image_impl.hpp
 * @author Mehul Kumar Nirala
 * @author Omar Shrit
 * @author Ryan Curtin
 * Implementation of image saving functionality via STB.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_IMAGE_IMPL_HPP
#define MLPACK_CORE_DATA_SAVE_IMAGE_IMPL_HPP

// In case it hasn't been included yet.
#include "save_image.hpp"

namespace mlpack {
namespace data {

// Image saving API for multiple files.
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageOptions& opts)
{
  if (files.empty())
  {
    std::stringstream oss;
    oss << "Save(): vector of image files is empty; nothing to save.";
    handleError(oss, opts);
  }

  for (size_t i = 0; i < files.size(); ++i)
  {
    // Check to see if the file type is supported.
    if (!opts.ImageFormatSupported(files.at(i), true))
    {
      std::stringstream oss;
      oss << "Save(): file type " << Extension(files.at(i))
          << " isn't supported. Currently image saving supports: ";
      for (auto extension : opts.SaveFileTypes())
        oss << "  " << extension;
      oss << "." << std::endl;
      handleError(oss, opts);
    }
  }

  size_t dimension = opts.Width() * opts.Height() * opts.Channels();
  // We only need to check the rows since it is a matrix.
  if (dimension != matrix.n_rows)
  {
    std::stringstream oss;
    oss << "data::Save(): The given image dimensions do not match the "
        << "dimensions of the matrix to be saved!";
    handleError(oss, opts);
  }

  bool success = false;
  for (size_t i = 0; i < files.size() ; ++i)
  {
    // I do not like the fact that we are looping over and over again and
    // trying to figure out the extension at evertime, usually the user have
    // all the of dataset of the same extension.
    // I also do not know how much this if else deduction is costing us, but
    // could be avoided since we are already looping and checking at the start
    // all of the provided extensions.
    if ("png" == Extension(files.at(i)))
    {
      success = stbi_write_png(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), matrix.colptr(i), opts.Width() * opts.Channels());
    }
    else if ("bmp" == Extension(files.at(i)))
    {
      success = stbi_write_bmp(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), matrix.colptr(i));
    }
    else if ("tga" == Extension(files.at(i)))
    {
      success = stbi_write_tga(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), matrix.colptr(i));
    }
    else if ("hdr" == Extension(files.at(i)))
    {
      // We have to convert to float for HDR.
      arma::fmat imageBuf = arma::conv_to<arma::fmat>::from(matrix.col(i));
      success = stbi_write_hdr(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), imageBuf.memptr());
    }
    else if ("jpg" == Extension(files.at(i)))
    {
      success = stbi_write_jpg(files.at(i).c_str(), opts.Width(), opts.Height(),
          opts.Channels(), matrix.colptr(i), opts.Quality());
    }

    if (!success)
    {
      std::stringstream oss;
      oss << "Save(): error saving image to '" << files.at(i) << "'.";
      handleError(oss, opts);
    }
  }
  return success;
}

} // namespace data
} // namespace mlpack

#endif
