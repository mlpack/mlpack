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

#ifdef HAS_STB // Compile this only if stb is present.

#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>

#if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include)
    && __has_include(<filesystem>)
#include <filesystem>
#define HAS_FILESYSTEM 1
namespace fs = std::filesystem;
#endif

#include "extension.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif

namespace mlpack {
namespace data {

/**
 * Loads a matrix with image. It also supports loading image from 
 * an entire directory.
 * @code
 * arma::Mat<unsigned char> img;
 * data::Image loader;
 * loader.Load("test_image.png", std::move(img));
 * @endcode
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
class Image
{
 public:
  /**
   * Image default constructor.
   */
  Image();

  /**
   * Instantiate the Image object with the image width, height, channels.
   *
   * @param width Matrix width for the output matrix.
   * @param height Matrix height for the output matrix.
   * @param channels Matrix channels for the output matrix.
   */
  Image(const size_t width, const size_t height, const size_t channels);

  /**
   * Checks if the given image filename is supported.
   *
   * @param filename Name of the image file.
   * @return Boolean value indicating success if it is an image.
   */
  bool ImageFormatSupported(const std::string& fileName, bool save = false);

  /**
   * Load the image file into the given matrix.
   *
   * @param fileName Name of the image file.
   * @param flipVertical Flip the image vertical upon loading.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool Load(const std::string& fileName,
            bool flipVertical,
            arma::Mat<unsigned char>&& outputMatrix);

  /**
   * Load the image file into the given matrix.
   *
   * @param fileName Name of the image file.
   * @param flipVertical Flip the image vertical upon loading.
   * @param width Width of the image file.
   * @param height Height of the image file.
   * @param channels Channels of the image file.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool Load(const std::string& fileName,
            bool flipVertical,
            arma::Mat<unsigned char>&& outputMatrix,
            size_t& width,
            size_t& height,
            size_t& channels);

  /**
   * Load the image file into the given matrix.
   *
   * @param files A vector containing names of the image file to be loaded.
   * @param flipVertical Flip the image vertical upon loading.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool Load(const std::vector<std::string>& files,
            bool flipVertical,
            arma::Mat<unsigned char>&& outputMatrix);

  /**
   * Saves the image file present in the given matrix.
   *
   * @param fileName Name of the image file.
   * @param width Width of the image that is being saved.
   * @param height Width of the image that is being saved.
   * @param channels Number of channels in the image that is being saved.
   * @param flipVertical Flip the image vertical before saving.
   * @param inputMatrix Matrix to save images from.
   * @return Boolean value indicating success or failure.
   */
  bool Save(const std::string& fileName,
            size_t width,
            size_t height,
            size_t channels,
            bool flipVertical,
            arma::Mat<unsigned char>&& outputMatrix,
            size_t quality = 95);

  /**
   * Saves the image file present in the given matrix.
   *
   * @param files A vector containing names of the image file to be saved.
   * @param width Width of the image that is being saved.
   * @param height Width of the image that is being saved.
   * @param channels Number of channels in the image that is being saved.
   * @param flipVertical Flip the image vertical before saving.
   * @param inputMatrix Matrix to save images from.
   * @return Boolean value indicating success or failure.
   */
  bool Save(const std::vector<std::string>& files,
            size_t width,
            size_t height,
            size_t channels,
            bool flipVertical,
            arma::Mat<unsigned char>&& inputMatrix);

  /**
   * Load the image file into the given matrix.
   *
   * @param dirPath Path containing the image files.
   * @param flipVertical Flip the image vertical upon loading.
   * @param outputMatrix Matrix to load into.
   * @return Boolean value indicating success or failure of load.
   */
  bool LoadDir(const std::string& dirPath,
               bool flipVertical,
               arma::Mat<unsigned char>&& outputMatrix);

  /**
   * Image default destructor.
   */
  ~Image();

 private:
  // To store supported image types that can be loaded.
  static const std::vector<std::string> loadFileTypes;

  // To store supported image types that can be saved.
  static const std::vector<std::string> saveFileTypes;

  // To store maxWidth.
  size_t maxWidth;

  // To store maxHeight.
  size_t maxHeight;

  // To store channels.
  size_t channels;
};

} // namespace data
} // namespace mlpack

// Include implementation of Image.
#include "load_image_impl.hpp"

#endif // HAS_STB.

#endif
