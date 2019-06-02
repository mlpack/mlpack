/**
 * @file load_image_impl.hpp
 * @author Mehul Kumar Nirala
 *
 * An image loading utillity
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

LoadImage::LoadImage():
        matrixWidth(0),
        matrixHeight(0),
        channels(3)
{
  fileTypes.clear();

  // Declare supported file types.
  fileTypes.insert(fileTypes.end(),
    {"jpg", "png", "tga", "bmp", "psd", "gif", "hdr", "pic", "pnm"});
}

LoadImage::LoadImage(size_t width,
          size_t height,
          size_t channels):
          matrixWidth(width),
          matrixHeight(height),
          channels(channels)
{
  fileTypes.clear();

  // Declare supported file types.
  fileTypes.insert(fileTypes.end(),
    {"jpg", "png", "tga", "bmp", "psd", "gif", "hdr", "pic", "pnm"});
}

LoadImage::~LoadImage()
{
  // Do nothing.
}

bool LoadImage::ImageFormatSupported(const std::string& fileName)
{
  // Iterate over all supported file types.
  for (auto extension : fileTypes)
    if (extension == Extension(fileName))
      return true;
  return false;
}

bool LoadImage::Load(const std::string& fileName,
                      bool flipVertical,
                      arma::Mat<unsigned char>&& outputMatrix,
                      size_t *width,
                      size_t *height,
                      size_t *channels)
{
  unsigned char *image;

  if (!ImageFormatSupported(fileName))
  {
    std::ostringstream oss;
    oss << "File type " << Extension(fileName) << " not supported.\n";
    oss << "Currently it supports ";
    for (auto extension : fileTypes)
      oss << " " << extension;
    oss << std::endl;
    throw std::runtime_error(oss.str());
    return false;
  }

  stbi_set_flip_vertically_on_load(flipVertical);

  // Temporary variablesneeded as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;

  // For grayscale images
  if (*channels == 1)
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
  *width = tempWidth;
  *height = tempHeight;
  *channels = tempChannels;
  size_t size = (*width)*(*height)*(*channels);

  // Copy memory location into armadillo Mat.
  outputMatrix = arma::Mat<unsigned char>(image, 1, size, false, true);
  return true;
}

bool LoadImage::Load(const std::string& fileName,
                    bool flipVertical,
                    arma::Mat<unsigned char>&& outputMatrix)
{
  size_t width, height;
  bool status = Load(fileName,
      flipVertical,
      std::move(outputMatrix),
      &width, &height, &channels);
  if (!status)
    return status;

  Log::Info << width <<" "<< height <<" "<< channels << std::endl;

  // Throw error if the image is incompatible with the matrix.
  if (matrixWidth > 0 && matrixHeight > 0 &&
     (width != matrixWidth || height != matrixHeight))
  {
    std::ostringstream oss;
    oss << "Image '" << fileName;
    oss << "' does not match matrix height or width." << std::endl;
    oss << "Image width: " << width;
    oss << ", Image height: " << height << std::endl;
    oss << "Matrix Width: " << matrixWidth;
    oss << ", Matrix Height: " << matrixHeight << std::endl;
    throw std::runtime_error(oss.str());
  }
  return status;
}

bool LoadImage::Load(const std::vector<std::string>& files,
                     bool flipVertical,
                     arma::Mat<unsigned char>&& outputMatrix)
{
  if (files.size() == 0)
  {
    std::ostringstream oss;
    oss << "File vector is empty." << std::endl;
    throw std::runtime_error(oss.str());
  }

  arma::Mat<unsigned char> img;
  size_t width, height;
  bool status = Load(files[0],
                     flipVertical,
                     std::move(img),
                     &width, &height, &channels);
  Log::Info << "Loaded " << files[0] << std::endl;

  // Decide matrix dimension using the image height and width.
  matrixWidth = std::max(matrixWidth, width);
  matrixHeight = std::max(matrixHeight, height);

  outputMatrix.set_size(files.size(), matrixWidth*matrixHeight*channels);
  outputMatrix.row(0) = img;

  for (size_t i = 1; i < files.size() ; i++)
  {
    status &= Load(files[i],
                   flipVertical,
                   std::move(outputMatrix.row(i)));
    Log::Info << "Loaded " << files[i] << std::endl;
  }
  return status;
}

bool LoadImage::LoadDIR(const std::string& dirPath,
                        bool flipVertical,
                        arma::Mat<unsigned char>&& outputMatrix)
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
  return Load(files, flipVertical, std::move(outputMatrix));
#else
  std::ostringstream oss;
  oss << "Filesystem not supported" << std::endl;
  oss << "Failed to include <filesystem> " << std::endl;
  throw std::runtime_error(oss.str());
  return false;
#endif
}

} // namespace data
} // namespace mlpack

#endif
