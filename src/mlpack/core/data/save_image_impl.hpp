/**
 * @file core/data/save_image_impl.hpp
 * @author Mehul Kumar Nirala
 *
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
#include "image_info.hpp"

namespace mlpack {
namespace data {

/**
 * Save the given image to the given filename.
 *
 * @param filename Filename to save to.
 * @param matrix Matrix containing image to be saved.
 * @param info Information about the image (width/height/channels/etc.).
 * @param fatal Whether an exception should be thrown on save failure.
 */
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal)
{
  arma::Mat<unsigned char> tmpMatrix =
      arma::conv_to<arma::Mat<unsigned char>>::from(matrix);

  return SaveImage(filename, tmpMatrix, info, fatal);
}

// Image saving API for multiple files.
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal)
{
  if (files.size() == 0)
  {
    if (fatal)
    {
      Log::Fatal << "Save(): vector of image files is empty; nothing to save."
          << std::endl;
    }
    else
    {
      Log::Warn << "Save(): vector of image files is empty; nothing to save."
          << std::endl;
    }

    return false;
  }

  arma::Mat<unsigned char> img;
  bool status = true;

  for (size_t i = 0; i < files.size() ; ++i)
  {
    arma::Mat<eT> colImg(matrix.colptr(i), matrix.n_rows, 1,
        false, true);
    status &= Save(files[i], colImg, info, fatal);
  }

  return status;
}

inline bool SaveImage(const std::string& filename,
                      arma::Mat<unsigned char>& image,
                      ImageInfo& info,
                      const bool fatal)
{
  // Check to see if the file type is supported.
  if (!ImageFormatSupported(filename, true))
  {
    std::ostringstream oss;
    oss << "Save(): file type " << Extension(filename) << " not supported.\n";
    oss << "Currently image saving supports ";
    for (auto extension : SaveFileTypes())
      oss << "  " << extension;
    oss << "." << std::endl;

    if (fatal)
    {
      Log::Fatal << oss.str();
    }
    else
    {
      Log::Warn << oss.str();
    }

    return false;
  }

  // Ensure the shape of the matrix is correct.
  if (image.n_cols > 1)
  {
    Log::Warn << "Save(): given input image matrix contains more than 1 image."
        << std::endl;
    Log::Warn << "Only the first image will be saved!" << std::endl;
  }

  if (info.Width() * info.Height() * info.Channels() != image.n_elem)
  {
    Log::Fatal << "data::Save(): The given image dimensions do not match the "
        << "dimensions of the matrix to be saved!" << std::endl;
  }

  bool status = false;
  unsigned char* imageMem = image.memptr();

  if ("png" == Extension(filename))
  {
    status = stbi_write_png(filename.c_str(), info.Width(), info.Height(),
        info.Channels(), imageMem, info.Width() * info.Channels());
  }
  else if ("bmp" == Extension(filename))
  {
    status = stbi_write_bmp(filename.c_str(), info.Width(), info.Height(),
        info.Channels(), imageMem);
  }
  else if ("tga" == Extension(filename))
  {
    status = stbi_write_tga(filename.c_str(), info.Width(), info.Height(),
        info.Channels(), imageMem);
  }
  else if ("hdr" == Extension(filename))
  {
    // We'll have to convert to float...
    arma::fmat tmpImage = arma::conv_to<arma::fmat>::from(image);
    status = stbi_write_hdr(filename.c_str(), info.Width(), info.Height(),
        info.Channels(), tmpImage.memptr());
  }
  else if ("jpg" == Extension(filename))
  {
    status = stbi_write_jpg(filename.c_str(), info.Width(), info.Height(),
        info.Channels(), imageMem, info.Quality());
  }

  if (!status)
  {
    if (fatal)
    {
      Log::Fatal << "Save(): error saving image to '" << filename << "'."
          << std::endl;
    }
    else
    {
      Log::Warn << "Save(): error saving image to '" << filename << "'."
          << std::endl;
    }
  }

  return status;
}

} // namespace data
} // namespace mlpack

#endif
