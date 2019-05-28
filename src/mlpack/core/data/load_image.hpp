/**
 * @file load_image.hpp
 * @author Mehul Kumar Nirala
 *
 * An image loading utillity
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_LOAD_IMAGE_HPP
#define MLPACK_CORE_DATA_LOAD_IMAGE_HPP

#include <string>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>
#include <boost/filesystem.hpp>

#include "extension.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace mlpack {
namespace data {

/**
 * Loads a matrix with image. It also supports loading image from 
 * an entire directory.
 *
 * The supported types of files are:
 * 
 * - JPEG baseline & progressive (12 bpc/arithmetic not supported,
 *    same as stock IJG lib)
 * - PNG 1/2/4/8/16-bit-per-channel
 * - TGA (not sure what subset, if a subset)
 * - BMP non-1bpp, non-RLE
 * - PSD (composited view only, no extra channels, 8/16 bit-per-channel)
 * - GIF (*comp always reports as 4-channel)
 * - HDR (radiance rgbE format)
 * - PIC (Softimage PIC)
 * - PNM (PPM and PGM binary only).
 */

class LoadImage
{
 public:
  LoadImage();

  /**
   * LoadImage constructor.
   *
   * @param width Matrix width for output matrix.
   * @param height Matrix height for output matrix.
   * @param channels Matrix channels for output matrix.
   */
  LoadImage(int width, int height, int channels);

  /**
   * Checks if the given image filename is supported.
   *
   * @param filename Name of the image file.
   * @return Boolean value indicating success if it is an image.
   */
  bool isImageFile(std::string fileName);


  /**
   * Load the image file into the given matrix.
   * Throws exceptions on errors.
   *
   * @param fileName Name of the image file.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool Load(std::string fileName, arma::Mat<unsigned char>&& outputMatrix);

  /**
   * Load the image file into the given matrix.
   * Throws exceptions on errors.
   *
   * @param fileName Name of the image file.
   * @param width Width of the image file.
   * @param height Height of the image file.
   * @param channels Channels of the image file.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool Load(std::string fileName,
          arma::Mat<unsigned char>&& outputMatrix,
          int *width,
          int *height,
          int *channels);

  /**
   * Load the image file into the given matrix.
   * Throws exceptions on errors.
   *
   * @param files A vector containing names of the image file to be loaded.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool Load(std::vector<std::string>& files,
    arma::Mat<unsigned char>&& outputMatrix);

  /**
   * Load the image file into the given matrix.
   * Throws exceptions on errors.
   *
   * @param dirPath Path containing the image files.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool LoadDIR(std::string dirPath, arma::Mat<unsigned char>&& outputMatrix);

  ~LoadImage();

 private:
  // To store supported image types.
  std::vector<std::string> fileTypes;

  // To store matrixWidth.
  int matrixWidth;

  // To store matrixHeight.
  int matrixHeight;

  // To store channels.
  int channels;
};

} // namespace data
} // namespace mlpack

// Include implementation of LoadImage.
#include "load_image_impl.hpp"

#endif
