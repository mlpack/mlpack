/**
 * @file core/data/audio_options.hpp
 * @author Omar Shrit
 *
 * Audio options, all possible options extracted from loading audio formats.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_AUDIO_OPTIONS_HPP
#define MLPACK_CORE_DATA_AUDIO_OPTIONS_HPP

#include <mlpack/prereqs.hpp>
#include "extension.hpp"
#include "data_options.hpp"

namespace mlpack {
class AudioOptions : public DataOptionsBase<AudioOptions>
{
 public:
  AudioOptions(std::optional<size_t> channels = std::nullopt,
               std::optional<size_t> sampleRate = std::nullopt,
               std::optional<size_t> bitsPerSample = std::nullopt,
               std::optional<size_t> audioDuration = std::nullopt,
               std::optional<size_t> totalPCMFramesCount = std::nullopt,
               std::optional<size_t> totalFramesRead = std::nullopt,
               std::optional<size_t> totalSamples = std::nullopt,
               std::optional<size_t> fileBitRate = std::nullopt) :
    DataOptionsBase<AudioOptions>(),
    channels(channels),
    sampleRate(sampleRate),
    bitPerSample(bitsPerSample),
    audioDuration(audioDuration),
    totalPCMFramesCount(totalPCMFramesCount),
    totalFramesRead(totalFramesRead),
    totalSamples(totalSamples),
    fileBitRate(fileBitRate)
  {
    // Do nothing.
  }

  AudioOptions(const DataOptions& opts) :
      DataOptionsBase<AudioOptions>(opts)
  {
    // Do nothing.
  }

  AudioOptions(const DataOptionsBase<AudioOptions>& opts) :
      DataOptionsBase<AudioOptions>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  AudioOptions(DataOptionsBase<AudioOptions>&& opts) :
      DataOptionsBase<AudioOptions>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  AudioOptions& operator=(const DataOptionsBase<AudioOptions>& otherIn)
  {
    const AudioOptions& other = static_cast<const AudioOptions&>(otherIn);

    if (&other == this)
      return *this;

    audioDuration       = other.audioDuration;
    bitPerSample        = other.bitPerSample;
    channels            = other.channels;
    fileBitRate         = other.fileBitRate;
    sampleRate          = other.sampleRate;
    totalFramesRead     = other.totalFramesRead;
    totalPCMFramesCount = other.totalPCMFramesCount;
    totalSamples        = other.totalSamples;

    // Copy base members.
    DataOptionsBase<AudioOptions>::operator=(other);

    return *this;
  }

  AudioOptions& operator=(DataOptionsBase<AudioOptions>&& otherIn)
  {
    AudioOptions&& other = static_cast<AudioOptions&&>(otherIn);

    if (&other == this)
      return *this;

    audioDuration       = std::move(other.audioDuration);
    bitPerSample        = std::move(other.bitPerSample);
    channels            = std::move(other.channels);
    fileBitRate         = std::move(other.fileBitRate);
    sampleRate          = std::move(other.sampleRate);
    totalFramesRead     = std::move(other.totalFramesRead);
    totalPCMFramesCount = std::move(other.totalPCMFramesCount);
    totalSamples        = std::move(other.totalSamples);

    // Move base members.
    DataOptionsBase<AudioOptions>::operator=(std::move(other));

    return *this;
  }

  // Conversions must be explicit.
  template<typename Derived2>
  explicit AudioOptions(const DataOptionsBase<Derived2>& other) :
      DataOptionsBase<AudioOptions>(other) { }

  template<typename Derived2>
  explicit AudioOptions(DataOptionsBase<Derived2>&& other) :
      DataOptionsBase<AudioOptions>(std::move(other)) { }

  template<typename Derived2>
  AudioOptions& operator=(const DataOptionsBase<Derived2>& other)
  {
    return static_cast<AudioOptions&>(
        DataOptionsBase<AudioOptions>::operator=(other));
  }

  template<typename Derived2>
  AudioOptions& operator=(DataOptionsBase<Derived2>&& other)
  {
    return static_cast<AudioOptions&>(
        DataOptionsBase<AudioOptions>::operator=(std::move(other)));
  }

  void Combine(const AudioOptions& other)
  {
    if (!audioDuration.has_value() && other.audioDuration.has_value())
    {
      audioDuration = other.audioDuration;
    }
    else if (audioDuration.has_value() && other.audioDuration.has_value())
    {
      if (audioDuration.value() != other.audioDuration.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "audioDuration with different values!");
      }
    }

    if (!bitPerSample.has_value() && other.bitPerSample.has_value())
    {
      bitPerSample = other.bitPerSample;
    }
    else if (bitPerSample.has_value() && other.bitPerSample.has_value())
    {
      if (bitPerSample.value() != other.bitPerSample.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "bitPerSample with different values!");
      }
    }

    if (!channels.has_value() && other.channels.has_value())
    {
      channels = other.channels;
    }
    else if (channels.has_value() && other.channels.has_value())
    {
      if (channels.value() != other.channels.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "channels with different values!");
      }
    }

    if (!fileBitRate.has_value() && other.fileBitRate.has_value())
    {
      fileBitRate = other.fileBitRate;
    }
    else if (fileBitRate.has_value() && other.fileBitRate.has_value())
    {
      if (fileBitRate.value() != other.fileBitRate.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "fileBitRate with different values!");
      }
    }

    if (!sampleRate.has_value() && other.sampleRate.has_value())
    {
      sampleRate = other.sampleRate;
    }
    else if (sampleRate.has_value() && other.sampleRate.has_value())
    {
      if (sampleRate.value() != other.sampleRate.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "sampleRate with different values!");
      }
    }

    if (!totalFramesRead.has_value() && other.totalFramesRead.has_value())
    {
      totalFramesRead = other.totalFramesRead;
    }
    else if (totalFramesRead.has_value() && other.totalFramesRead.has_value())
    {
      if (totalFramesRead.value() != other.totalFramesRead.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "totalFramesRead with different values!");
      }
    }

    if (!totalPCMFramesCount.has_value() &&
        other.totalPCMFramesCount.has_value())
    {
      totalPCMFramesCount = other.totalPCMFramesCount;
    }
    else if (totalPCMFramesCount.has_value() &&
        other.totalPCMFramesCount.has_value())
    {
      if (totalPCMFramesCount.value() !=
          other.totalPCMFramesCount.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "totalPCMFramesCount with different values!");
      }
    }

    if (!totalSamples.has_value() && other.totalSamples.has_value())
    {
      totalSamples = other.totalSamples;
    }
    else if (totalSamples.has_value() && other.totalSamples.has_value())
    {
      if (totalSamples.value() != other.totalSamples.value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "totalSamples with different values!");
      }
    }
  }

  void WarnBaseConversion(const char* dataDescription) const
  {
    if (audioDuration.has_value() && audioDuration != defaultAudioDuration)
      this->WarnOptionConversion("audioDuration", dataDescription);
    if (bitPerSample.has_value() && bitPerSample != defaultBitPerSample)
      this->WarnOptionConversion("bitPerSample", dataDescription);
    if (channels.has_value() && channels != defaultChannels)
      this->WarnOptionConversion("channels", dataDescription);
    if (fileBitRate.has_value() && fileBitRate != defaultFileBitRate)
      this->WarnOptionConversion("fileBitRate", dataDescription);
    if (sampleRate.has_value() && sampleRate != defaultSampleRate)
      this->WarnOptionConversion("sampleRate", dataDescription);
    if (totalFramesRead.has_value() &&
        totalFramesRead != defaultTotalFramesRead)
      this->WarnOptionConversion("totalFramesRead", dataDescription);
    if (totalPCMFramesCount.has_value() &&
        totalPCMFramesCount != defaultTotalPCMFramesCount)
      this->WarnOptionConversion("totalPCMFramesCount", dataDescription);
    if (totalSamples.has_value() && totalSamples != defaultTotalSamples)
      this->WarnOptionConversion("totalSamples", dataDescription);
  }

  static const char* DataDescription() { return "audio data"; }

  void Reset()
  {
    audioDuration.reset();
    bitPerSample.reset();
    channels.reset();
    fileBitRate.reset();
    sampleRate.reset();
    totalFramesRead.reset();
    totalPCMFramesCount.reset();
    totalSamples.reset();
  }

  size_t AudioDuration() const
  {
    return this->AccessMember(audioDuration, defaultAudioDuration);
  }

  size_t& AudioDuration()
  {
    return this->ModifyMember(audioDuration, defaultAudioDuration);
  }

  size_t BitsPerSample() const
  {
    return this->AccessMember(bitPerSample, defaultBitPerSample);
  }

  size_t& BitsPerSample()
  {
    return this->ModifyMember(bitPerSample, defaultBitPerSample);
  }

  size_t Channels() const
  {
    return this->AccessMember(channels, defaultChannels);
  }

  size_t& Channels()
  {
    return this->ModifyMember(channels, defaultChannels);
  }

  size_t FileBitRate() const
  {
    return this->AccessMember(fileBitRate, defaultFileBitRate);
  }

  size_t& FileBitRate()
  {
    return this->ModifyMember(fileBitRate, defaultFileBitRate);
  }

  size_t SampleRate() const
  {
    return this->AccessMember(sampleRate, defaultSampleRate);
  }

  size_t& SampleRate()
  {
    return this->ModifyMember(sampleRate, defaultSampleRate);
  }

  size_t TotalFramesRead() const
  {
    return this->AccessMember(totalFramesRead, defaultTotalFramesRead);
  }

  size_t& TotalFramesRead()
  {
    return this->ModifyMember(totalFramesRead, defaultTotalFramesRead);
  }

  size_t TotalPCMFramesCount() const
  {
    return this->AccessMember(totalPCMFramesCount,
        defaultTotalPCMFramesCount);
  }

  size_t& TotalPCMFramesCount()
  {
    return this->ModifyMember(totalPCMFramesCount,
        defaultTotalPCMFramesCount);
  }

  size_t TotalSamples() const
  {
    return this->AccessMember(totalSamples, defaultTotalSamples);
  }

  size_t& TotalSamples()
  {
    return this->ModifyMember(totalSamples, defaultTotalSamples);
  }

 private:
  std::optional<size_t> channels;
  std::optional<size_t> sampleRate;
  std::optional<size_t> bitPerSample;
  std::optional<size_t> audioDuration;
  std::optional<size_t> totalPCMFramesCount;
  std::optional<size_t> totalFramesRead;
  std::optional<size_t> totalSamples;
  std::optional<size_t> fileBitRate;

  constexpr static const size_t defaultAudioDuration    = 0;
  constexpr static const size_t defaultBitPerSample     = 0;
  constexpr static const size_t defaultChannels         = 0;
  constexpr static const size_t defaultFileBitRate      = 0;
  constexpr static const size_t defaultSampleRate       = 0;
  constexpr static const size_t defaultTotalFramesRead  = 0;
  constexpr static const size_t defaultTotalPCMFramesCount = 0;
  constexpr static const size_t defaultTotalSamples     = 0;
};

template<>
struct IsDataOptions<AudioOptions>
{
  constexpr static bool value = true;
};

} // namespace mlpack

#endif
