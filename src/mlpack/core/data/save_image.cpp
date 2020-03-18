/**
 * @file save_image.cpp
 * @author Mehul Kumar Nirala
 *
 * Implementation of image saving functionality via STB.
 */
#include "save.hpp"

#ifdef HAS_STB

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace mlpack {
namespace data {

bool SaveImage(const std::string& filename,
               arma::Mat<unsigned char>& image,
               ImageInfo& info,
               const bool fatal,
               const bool transpose)
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

  stbi_flip_vertically_on_write(transpose);

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
               const bool fatal,
               const bool transpose)
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
