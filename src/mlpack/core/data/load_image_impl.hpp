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

#ifdef HAS_STB // Compile this only if stb is present.

// In case it hasn't been included yet.
#include "load_image.hpp"

namespace mlpack {
namespace data {

Image::Image():
        maxWidth(0),
        maxHeight(0),
        channels(3)
{
  // Do nothing.
}

Image::Image(const size_t width,
             const size_t height,
             const size_t channels):
             maxWidth(width),
             maxHeight(height),
             channels(channels)
{
  // Do nothing.
}

Image::~Image()
{
  // Do nothing.
}

bool Image::Load(const std::string& fileName,
                 arma::Mat<unsigned char>& outputMatrix,
                 size_t& width,
                 size_t& height,
                 size_t& channels,
                 bool flipVertical)
{
  unsigned char* image;

  if (!ImageFormatSupported(fileName))
  {
    std::ostringstream oss;
    oss << "File type " << Extension(fileName) << " not supported.\n";
    oss << "Currently it supports ";
    for (auto extension : loadFileTypes)
      oss << " " << extension;
    oss << std::endl;
    throw std::runtime_error(oss.str());
    return false;
  }

  stbi_set_flip_vertically_on_load(flipVertical);

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;

  // For grayscale images.
  if (channels == 1)
  {
    image = stbi_load(fileName.c_str(),
             &tempWidth,
             &tempHeight,
             &tempChannels,
             STBI_grey);
  }
  else
  {
    image = stbi_load(fileName.c_str(),
             &tempWidth,
             &tempHeight,
             &tempChannels,
             STBI_rgb);
  }

  if (tempWidth <= 0 || tempHeight <= 0)
  {
    std::ostringstream oss;
    oss << "Image '" << fileName << "' not found." << std::endl;
    free(image);
    throw std::runtime_error(oss.str());
    return false;
  }
  width = tempWidth;
  height = tempHeight;
  channels = tempChannels;
  size_t size = width * height * channels;

  // Copy image into armadillo Mat.
  outputMatrix = arma::Mat<unsigned char>(image, size, 1, true, true);

  // Free the image pointer.
  free(image);
  return true;
}

bool Image::Load(const std::string& fileName,
                 arma::Mat<unsigned char>& outputMatrix,
                 bool flipVertical)
{
  size_t width, height;
  bool status = Load(fileName, outputMatrix,
      width, height, channels, flipVertical);
  if (!status)
    return status;

  Log::Info << "Image width: " << width <<"  height: "<< height <<" channels: "
      << channels << std::endl;

  // Throw error if the image is incompatible with the matrix.
  if (maxWidth > 0 && maxHeight > 0 &&
     (width != maxWidth || height != maxHeight))
  {
    std::ostringstream oss;
    oss << "Image '" << fileName;
    oss << "' does not match matrix height or width." << std::endl;
    oss << "Image width: " << width;
    oss << ", Image height: " << height << std::endl;
    oss << "Max Width: " << maxWidth;
    oss << ", Max Height: " << maxHeight << std::endl;
    throw std::runtime_error(oss.str());
  }
  return status;
}

bool Image::Load(const std::vector<std::string>& files,
                 arma::Mat<unsigned char>& outputMatrix,
                 size_t& width,
                 size_t& height,
                 size_t& channels,
                 bool flipVertical)
{
  if (files.size() == 0)
  {
    std::ostringstream oss;
    oss << "File vector is empty." << std::endl;
    throw std::runtime_error(oss.str());
  }

  arma::Mat<unsigned char> img;

  bool status = Load(files[0], img, width,
      height, channels, flipVertical);
  Log::Info << "Loaded " << files[0] << std::endl;

  // Decide matrix dimension using the image height and width.
  maxWidth = std::max(maxWidth, width);
  maxHeight = std::max(maxHeight, height);

  outputMatrix.set_size(maxWidth * maxHeight * channels, files.size());
  outputMatrix.col(0) = img;

  for (size_t i = 1; i < files.size() ; i++)
  {
    arma::Mat<unsigned char> colImg(outputMatrix.colptr(i), outputMatrix.n_rows,
        1, false, true);
    status &= Load(files[i],
                   colImg,
                   flipVertical);
    Log::Info << "Loaded " << files[i] << std::endl;
  }
  return status;
}

bool Image::Save(const std::string& fileName,
                 arma::Mat<unsigned char>& inputMatrix,
                 size_t width,
                 size_t height,
                 size_t channels,
                 bool flipVertical,
                 size_t quality)
{
  if (!ImageFormatSupported(fileName, true))
  {
    std::ostringstream oss;
    oss << "File type " << Extension(fileName) << " not supported.\n";
    oss << "Currently it supports ";
    for (auto extension : saveFileTypes)
      oss << ", " << extension;
    oss << std::endl;
    throw std::runtime_error(oss.str());
    return false;
  }
  if (inputMatrix.n_cols > 1)
  {
    std::cout << "Input Matrix contains more than 1 image." << std::endl;
    std::cout << "Only the firstimage will be saved!" << std::endl;
  }
  unsigned char* image = inputMatrix.memptr();
  stbi_flip_vertically_on_write(flipVertical);

  bool status = false;
  int tempWidth = width, tempHeight = height, tempChannels = channels;
  if ("png" == Extension(fileName))
  {
    status = stbi_write_png(fileName.c_str(), tempWidth, tempHeight,
        tempChannels, image, width * channels);
  }
  else if ("bmp" == Extension(fileName))
  {
    status = stbi_write_bmp(fileName.c_str(), tempWidth, tempHeight,
        tempChannels, image);
  }
  else if ("tga" == Extension(fileName))
  {
    status = stbi_write_tga(fileName.c_str(), tempWidth, tempHeight,
        tempChannels, image);
  }
  else if ("hdr" == Extension(fileName))
  {
    status = stbi_write_hdr(fileName.c_str(), tempWidth, tempHeight,
        tempChannels, reinterpret_cast<float*>(image));
  }
  else if ("jpg" == Extension(fileName))
  {
    status = stbi_write_jpg(fileName.c_str(), tempWidth, tempHeight,
        tempChannels, image, quality);
  }
  return status;
}

bool Image::Save(const std::vector<std::string>& files,
                 arma::Mat<unsigned char>& inputMatrix,
                 size_t width,
                 size_t height,
                 size_t channels,
                 bool flipVertical,
                 size_t quality)
{
  bool status = true;
  for (size_t i = 0; i < files.size(); i++)
  {
    arma::Mat<unsigned char> colImg(inputMatrix.colptr(i), inputMatrix.n_rows,
        1, false, true);
    if (!Save(files[i], colImg, width, height, channels,
        flipVertical, quality))
    {
      std::cout << "Unable to save '" << files[i] << "'." << std::endl;
      status = false;
    }
  }
  return status;
}

bool Image::LoadDir(const std::string& dirPath,
                    arma::Mat<unsigned char>& outputMatrix,
                    bool flipVertical)
{
  std::vector<std::string> files;
#ifdef HAS_FILESYSTEM
  // cycle through the directory
  for (std::string& file : fs::filesystem::directory_iterator(dirPath))
  {
    // If it's not a directory, list it.
    if (fs::filesystem::is_regular_file(file)) {
      // Load only image files in the directory.
      if (ImageFormatSupported(file))
        files.push_back(file);
    }
  }
  return Load(files, outputMatrix, flipVertical);
#else
  std::ostringstream oss;
  oss << "C++17 filesystem library not available." << std::endl;
  oss << "Failed to include <filesystem> header!" << std::endl;
  throw std::runtime_error(oss.str());
  return false;
#endif
}

// Image loading API.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  std::vector<std::string> files{filename};
  return Load(files, matrix, info, fatal, transpose);
}

// Image loading API for multiple files.
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  Timer::Start("loading_image");
  bool status;
  try
  {
    Image loader;
    status = loader.Load(files, matrix, info.width, info.height,
        info.channels, info.flipVertical);

    // We transpose by default. So, un-transpose if necessary.
    if (!transpose)
      matrix = arma::trans(matrix);
  }
  catch (std::exception& e)
  {
    Timer::Stop("loading_image");
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }

  Timer::Start("loading_image");
  return status;
}

// Image saving API.
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  std::vector<std::string> files{filename};
  return Save(files, matrix, info, fatal, transpose);
}

// Image saving API for multiple files.
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  Timer::Start("saving_image");
  bool status;

  // We transpose by default. So, un-transpose if necessary.
  if (!transpose)
    matrix = arma::trans(matrix);

  try
  {
    Image saver;
    status = saver.Save(files, matrix, info.width, info.height,
        info.channels, info.flipVertical, info.quality);
  }
  catch (std::exception& e)
  {
    Timer::Stop("saving_image");
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }

  Timer::Start("saving_image");
  return status;
}

} // namespace data
} // namespace mlpack

#endif // HAS_STB.

#endif
