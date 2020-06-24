/**
 * @file core/data/save_image.cpp
 * @author Mehul Kumar Nirala
 *
 * Implementation of image saving functionality via STB.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "save.hpp"

#ifdef HAS_STB

// The implementation of the functions is included directly, so we need to make
// sure it doesn't get included twice.  This is to work around a bug in old
// versions of STB where not all functions were correctly marked static.
#define STB_IMAGE_WRITE_STATIC
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
  #define STB_IMAGE_WRITE_IMPLEMENTATION
#else
  #undef STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include <stb_image_write.h>
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
  #define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

namespace mlpack {
namespace data {

bool SaveImage(const std::string& filename,
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
    for (auto extension : saveFileTypes)
      oss << ", " << extension;
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

#else

namespace mlpack {
namespace data {

bool SaveImage(const std::string& /* filename */,
               arma::Mat<unsigned char>& /* image */,
               ImageInfo& /* info */,
               const bool fatal)
{
  if (fatal)
  {
    Log::Fatal << "Save(): mlpack was not compiled with STB support, so images "
        << "cannot be saved!" << std::endl;
  }
  else
  {
    Log::Warn << "Save(): mlpack was not compiled with STB support, so images "
        << "cannot be saved!" << std::endl;
  }

  return false;
}

} // namespace data
} // namespace mlpack

#endif
