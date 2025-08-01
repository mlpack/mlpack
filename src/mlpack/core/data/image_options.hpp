/**
 * @file core/data/image_options.hpp
 * @author Mehul Kumar Nirala
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Image options, all possible options to load different image formats
 * with specific settings into mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_INFO_HPP
#define MLPACK_CORE_DATA_IMAGE_INFO_HPP

#include <mlpack/prereqs.hpp>
#include "extension.hpp"

namespace mlpack {
namespace data {

/**
 * Implements meta-data of images required by data::Load and
 * data::Save for loading and saving images into arma::Mat.
 */
class ImageOptions : public DataOptionsBase<ImageOptions>
{
 public:
  /**
   * Instantiate the ImageOptions object with the given image width, height,
   * number of channels and quality parameter.
   *
   * @param width Image width.
   * @param height Image height.
   * @param channels Number of channels in the image.
   * @param quality Compression of the image if saved as jpg (0 - 100).
   */
  ImageOptions(const size_t width = 0,
               const size_t height = 0,
               const size_t channels = 3,
               const size_t quality = 90) :
    width(width),
    height(height),
    channels(channels),
    quality(quality)
  {
    // Do nothing.
  }

  ImageOptions(const DataOptionsBase<ImageOptions>& opts) :
      DataOptionsBase<ImageOptions>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  ImageOptions(DataOptionsBase<ImageOptions>&& opts) :
      DataOptionsBase<ImageOptions>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  ImageOptions& operator=(const DataOptionsBase<ImageOptions>& otherIn)
  {
    const ImageOptions& other = static_cast<const ImageOptions&>(otherIn);

    if (&other == this)
      return *this;

    width    = other.width;
    height   = other.height;
    channels = other.channels;
    quality  = other.quality;

    // Copy base members.
    DataOptionsBase<ImageOptions>::operator=(other);

    return *this;
  }

  ImageOptions& operator=(DataOptionsBase<ImageOptions>&& otherIn)
  {
    ImageOptions&& other = static_cast<ImageOptions&&>(otherIn);

    if (&other == this)
      return *this;

    width    = std::move(other.width);
    height   = std::move(other.height);
    channels = std::move(other.channels);
    quality  = std::move(other.quality);

    // Move base members.
    DataOptionsBase<ImageOptions>::operator=(std::move(other));

    return *this;
  }

  //
  // Handling for copy and move operations on other DataOptionsBase types.
  //

  // Conversions must be explicit.
  template<typename Derived2>
  explicit ImageOptions(const DataOptionsBase<Derived2>& other) :
      DataOptionsBase<ImageOptions>(other) { }

  template<typename Derived2>
  explicit ImageOptions(DataOptionsBase<Derived2>&& other) :
      DataOptionsBase<ImageOptions>(std::move(other)) { }

  template<typename Derived2>
  ImageOptions& operator=(const DataOptionsBase<Derived2>& other)
  {
    return static_cast<ImageOptions&>(
        DataOptionsBase<ImageOptions>::operator=(other));
  }

  template<typename Derived2>
  ImageOptions& operator=(DataOptionsBase<Derived2>&& other)
  {
    return static_cast<ImageOptions&>(
        DataOptionsBase<ImageOptions>::operator=(std::move(other)));
  }

  static const char* DataDescription() { return "image data"; }

  // @rcurtin do we really need this if private memebers are not
  // std::optionals ??
  void Reset()
  {
    width = 0;
    height = 0;
    channels = 3;
    quality = 90;
  }

  inline const std::vector<std::string> LoadFileTypes()
  {
    return std::vector<std::string>({"jpg", "png", "tga", "bmp", "psd", "gif",
        "hdr", "pic", "pnm", "jpeg"});
  }

  inline const std::vector<std::string> SaveFileTypes()
  {
    return std::vector<std::string>({"jpg", "png", "tga", "bmp", "hdr"});
  }

  /**
   * Checks if the given image filename is supported.
   *
   * @param fileName Name of the image file.
   * @param save Set to true to check if the file format can be saved, else loaded.
   * @return Boolean value indicating success if it is an image.
   */
  inline bool ImageFormatSupported(const std::string& fileName,
                                   const bool save = false)
  {
    if (save)
    {
      // Iterate over all supported file types that can be saved.
      for (auto extension : SaveFileTypes())
      {
        if (extension == Extension(fileName))
          return true;
      }
    }
    else
    {
      // Iterate over all supported file types that can be loaded.
      for (auto extension : LoadFileTypes())
      {
        if (extension == Extension(fileName))
          return true;
      }
    }

    return false;
  }

  //! Get the image width.
  const size_t& Width() const { return width; }
  //! Modify the image width.
  size_t& Width() { return width; }
  //! Get the image height.

  const size_t& Height() const { return height; }
  //! Modify the image height.
  size_t& Height() { return height; }

  //! Get the image channels.
  const size_t& Channels() const { return channels; }
  //! Modify the image channels.
  size_t& Channels() { return channels; }

  //! Get the image quality.
  const size_t& Quality() const { return quality; }
  //! Modify the image quality.
  size_t& Quality() { return quality; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(width));
    ar(CEREAL_NVP(channels));
    ar(CEREAL_NVP(height));
    ar(CEREAL_NVP(quality));
  }

 private:
  // To store the image width.
  size_t width;

  // To store the image height.
  size_t height;

  // To store the number of channels in the image.
  size_t channels;

  // Compression of the image if saved as jpg (0 - 100).
  size_t quality;

  inline static const std::unordered_set<std::string> saveType
      = {"jpg", "png", "tga", "bmp", "hdr"};

  inline static const std::unordered_set<std::string> loadType
      = {"jpg", "png", "tga", "bmp", "psd", "gif", "hdr", "pic", "pnm", "jpeg"};

};

// Provide backward compatibility with the previous API
// This should be removed with mlpack 5.0.0
using ImageInfo = ImageOptions;

} // namespace data
} // namespace mlpack

#endif
