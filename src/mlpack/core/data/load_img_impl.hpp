/**
 * @file load_img_impl.hpp
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
#include "load_img.hpp"

namespace mlpack {
namespace data {

LoadImage::LoadImage():
        matrixWidth(-1),
        matrixHeight(-1),
        channels(3)
{
  fileTypes.clear();

  // Declare supported file types.
  fileTypes.insert(fileTypes.end(),
    {"jpg","png","tga", "bmp", "psd", "gif", "hdr", "pic", "pnm"});
}

LoadImage::LoadImage(int width,
          int height,
          int channels):
          matrixWidth(width),
          matrixHeight(height),
          channels(channels)
{
  fileTypes.clear();

  // Declare supported file types.
  fileTypes.insert(fileTypes.end(),
    {"jpg","png","tga", "bmp", "psd", "gif", "hdr", "pic", "pnm"});
}

LoadImage::~LoadImage()
{
  // Do nothing.
}

bool LoadImage::isImageFile(std::string fileName)
{   
  // Iterate over all supported file types.
  for(auto extension: fileTypes){
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == Extension(fileName))
      return true;
  }
  return false;
}

void LoadImage::Load(std::string fileName, int *width, int *height, int *channels, arma::Mat<unsigned char>&& outputMatrix)
{
  unsigned char *image;
  char fileNameTemp[fileName.size() + 1];
  strcpy(fileNameTemp, fileName.c_str()); 
  stbi_set_flip_vertically_on_load(true);

  // For grayscale images
  if(*channels == 1)
  {
    image = stbi_load(fileNameTemp,
             width,
             height,
             channels,
             STBI_grey);
  }
  else
  {
    image = stbi_load(fileNameTemp,
             width,
             height,
             channels,
             STBI_rgb);
  }
  int size = (*width)*(*height)*(*channels);

  // Copy memory location into armadillo Mat.
  outputMatrix = arma::Mat<unsigned char>(image, 1, size, false, true );
}

void LoadImage::Load(std::string fileName, arma::Mat<unsigned char>&& outputMatrix)
{
  int width, height;
  Load(fileName, &width, &height, &channels, std::move(outputMatrix));
  Log::Info << width <<" "<< height <<" "<< channels << std::endl << std::flush;

  // Throw error if the image is incompatible with the matrix.
  if (width != matrixWidth || height != matrixHeight)
  {
    std::ostringstream oss;
    oss << "Image '" << fileName << "' does not match matrix height or width" << std::endl;
    oss << "Image width: " << width << ", Image height: " << height << std::endl;
    oss << "Matrix Width: " << matrixWidth << ", Matrix Height: " << matrixHeight << std::endl;
    throw std::runtime_error(oss.str());
  }
}

void LoadImage::Load(std::vector<std::string>& files, arma::Mat<unsigned char>&& outputMatrix)
{
  if (files.size() < 1)
  {
    std::ostringstream oss;
    oss << "File vector is empty." << std::endl;
    throw std::runtime_error(oss.str());
  }

  arma::Mat<unsigned char> img;
  int width, height;
  Load(files[0], &width, &height, &channels, std::move(img));
  Log::Info << "Loaded " << files[0] << std::endl << std::flush;

  // Decide matrix dimension using the image height and width.
  matrixWidth = std::max(matrixWidth, width);
  matrixHeight = std::max(matrixHeight, height);

  outputMatrix.set_size(files.size(), matrixWidth*matrixHeight*channels);
  outputMatrix.row(0) = img;

  for (size_t i = 1; i < files.size() ; i++)
  {
    Load(files[i], std::move(outputMatrix.row(i)));
    Log::Info << "Loaded " << files[i] << std::endl << std::flush;
  }
}

void LoadImage::LoadDir(std::string dirPath, arma::Mat<unsigned char>&& outputMatrix)
{
  boost::filesystem::path p (dirPath);

  boost::filesystem::directory_iterator end_itr;

  std::vector<std::string> files;

  // cycle through the directory
  for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr)
  {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (boost::filesystem::is_regular_file(itr->path())) {
      // assign current file name to current_file and echo it out to the console.
      std::string current_file = itr->path().string();
      // Load only image files in the directory.
      if (isImageFile(current_file))
        files.push_back(current_file);

    }
  }

  Load(files, std::move(outputMatrix));
}

} // namespace data
} // namespace mlpack

#endif
