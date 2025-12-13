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
#include "data_options.hpp"

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
   * #param image Indicate if we are loading / saving an image. 
   */
  ImageOptions(std::optional<size_t> width = std::nullopt,
               std::optional<size_t> height = std::nullopt,
               std::optional<size_t> channels = std::nullopt,
               std::optional<size_t> quality = std::nullopt) :
    DataOptionsBase<ImageOptions>(),
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

  void Combine(const ImageOptions& other)
  {
    if (!width.has_value() && other.width.has_value())
    {
      width = other.width;
    }
    else if (width.has_value() && other.width.has_value())
    {
      if (width.has_value() != other.width.has_value())
      {
        throw std::invalid_argument("ImageOptions: operator+(): cannot combine"
            "width with different values!");
      }
    }

    if (!height.has_value() && other.height.has_value())
    {
      height = other.height;
    }
    else if (height.has_value() && other.height.has_value())
    {
      if (height.has_value() != other.height.has_value())
      {
        throw std::invalid_argument("ImageOptions: operator+(): cannot combine"
            "height with different values!");
      }
    }

    if (!channels.has_value() && other.channels.has_value())
    {
      channels = other.channels;
    }
    else if (channels.has_value() && other.channels.has_value())
    {
      if (channels.has_value() != other.channels.has_value())
      {
        throw std::invalid_argument("ImageOptions: operator+(): cannot combine"
            "channels with different values!");
      }
    }

    if (!quality.has_value() && other.quality.has_value())
    {
      quality = other.quality;
    }
    else if (quality.has_value() && other.quality.has_value())
    {
      if (quality.has_value() != other.quality.has_value())
      {
        throw std::invalid_argument("ImageOptions: operator+(): cannot combine"
            "quality with different values!");
      }
    }
  }

  // Print warnings for any members that cannot be represented by a
  // DataOptionsBase<void>.
  void WarnBaseConversion(const char* dataDescription) const
  {
    if (width.has_value() && width != defaultWidth)
      this->WarnOptionConversion("width", dataDescription);
    if (height.has_value() && height != defaultHeight)
      this->WarnOptionConversion("height", dataDescription);
    if (channels.has_value() && channels != defaultChannels)
      this->WarnOptionConversion("channels", dataDescription);
    if (quality.has_value() && quality != defaultQuality)
      this->WarnOptionConversion("quality", dataDescription);
  }

  static const char* DataDescription() { return "image data"; }

  void Reset()
  {
    width.reset();
    height.reset();
    channels.reset();
    quality.reset();
  }

  size_t Width() const { return this->AccessMember(width, defaultWidth); }
  size_t& Width() { return this->ModifyMember(width, defaultWidth); }

  size_t Height() const { return this->AccessMember(height, defaultHeight); }
  size_t& Height() { return this->ModifyMember(height, defaultHeight); }

  size_t Channels() const
  {
    return this->AccessMember(channels, defaultChannels);
  }

  size_t& Channels() { return this->ModifyMember(channels, defaultChannels); }

  size_t Quality() const { return this->AccessMember(quality, defaultQuality); }
  size_t& Quality() { return this->ModifyMember(quality, defaultQuality); }

  inline static const std::unordered_set<std::string> saveType
      = {"jpg", "png", "tga", "bmp"};

  inline static const std::unordered_set<std::string> loadType
      = {"jpg", "png", "tga", "bmp", "psd", "gif", "pic", "pnm", "jpeg"};

 private:
  std::optional<size_t> width;
  std::optional<size_t> height;
  std::optional<size_t> channels;
  std::optional<size_t> quality;

  constexpr static const size_t defaultWidth = 0;
  constexpr static const size_t defaultHeight = 0;
  constexpr static const size_t defaultChannels = 3;
  constexpr static const size_t defaultQuality = 90;
};

template<>
struct IsDataOptions<ImageOptions>
{
  constexpr static bool value = true;
};

// Provide backward compatibility with the previous API
// This should be removed with mlpack 5.0.0
using ImageInfo = ImageOptions;

} // namespace data
} // namespace mlpack

#endif
