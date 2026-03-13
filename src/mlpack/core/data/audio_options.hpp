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

  AudioOptions()
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
    avgBytesPerSec      = other.avgBytesPerSec;
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
    avgBytesPerSec      = std::move(other.avgBytesPerSec);
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
      if (audioDuration.has_value() != other.audioDuration.has_value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "audioDuration with different values!");
      }
    }

    if (!avgBytesPerSec.has_value() && other.avgBytesPerSec.has_value())
    {
      avgBytesPerSec = other.avgBytesPerSec;
    }
    else if (avgBytesPerSec.has_value() && other.avgBytesPerSec.has_value())
    {
      if (avgBytesPerSec.has_value() != other.avgBytesPerSec.has_value())
      {
        throw std::invalid_argument("AudioOptions: operator+(): cannot combine"
            "avgBytesPerSec with different values!");
      }
    }

    if (!bitPerSample.has_value() && other.bitPerSample.has_value())
    {
      bitPerSample = other.bitPerSample;
    }
    else if (bitPerSample.has_value() && other.bitPerSample.has_value())
    {
      if (bitPerSample.has_value() != other.bitPerSample.has_value())
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
      if (channels.has_value() != other.channels.has_value())
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
      if (fileBitRate.has_value() != other.fileBitRate.has_value())
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
      if (sampleRate.has_value() != other.sampleRate.has_value())
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
      if (totalFramesRead.has_value() != other.totalFramesRead.has_value())
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
      if (totalPCMFramesCount.has_value() !=
          other.totalPCMFramesCount.has_value())
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
      if (totalSamples.has_value() != other.totalSamples.has_value())
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
    if (avgBytesPerSec.has_value() && avgBytesPerSec != defaultAvgBytesPerSec)
      this->WarnOptionConversion("avgBytesPerSec", dataDescription);
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
    avgBytesPerSec.reset();
    bitPerSample.reset();
    channels.reset();
    fileBitRate.reset();
    sampleRate.reset();
    totalFramesRead.reset();
    totalPCMFramesCount.reset();
    totalSamples.reset();
  }

  drwav_uint64 AudioDuration() const
  {
    return this->AccessMember(audioDuration, defaultAudioDuration);
  }

  drwav_uint64& AudioDuration()
  {
    return this->ModifyMember(audioDuration, defaultAudioDuration);
  }

  drwav_uint32 AvgBytesPerSec() const
  {
    return this->AccessMember(avgBytesPerSec, defaultAvgBytesPerSec);
  }

  drwav_uint32& AvgBytesPerSec()
  {
    return this->ModifyMember(avgBytesPerSec, defaultAvgBytesPerSec);
  }

  // @rcurtin Not available on MP3, only WAV. How we can expose this nicely?
  drwav_uint16 BitsPerSample() const
  {
    return this->AccessMember(bitPerSample, defaultBitPerSample);
  }

  // @rcurtin Not available on MP3, only WAV. How we can expose this nicely?
  drwav_uint16& BitsPerSample()
  {
    return this->ModifyMember(bitPerSample, defaultBitPerSample);
  }

  drwav_uint16 Channels() const
  {
    return this->AccessMember(channels, defaultChannels);
  }

  drwav_uint16& Channels()
  {
    return this->ModifyMember(channels, defaultChannels);
  }

  drwav_uint64 FileBitRate() const
  {
    return this->AccessMember(fileBitRate, defaultFileBitRate);
  }

  drwav_uint64& FileBitRate()
  {
    return this->ModifyMember(fileBitRate, defaultFileBitRate);
  }

  drwav_uint32 SampleRate() const
  {
    return this->AccessMember(sampleRate, defaultSampleRate);
  }

  drwav_uint32& SampleRate()
  {
    return this->ModifyMember(sampleRate, defaultSampleRate);
  }

  drwav_uint64 TotalFramesRead() const
  {
    return this->AccessMember(totalFramesRead, defaultTotalFramesRead);
  }

  drwav_uint64& TotalFramesRead()
  {
    return this->ModifyMember(totalFramesRead, defaultTotalFramesRead);
  }

  drwav_uint64 TotalPCMFramesCount() const
  {
    return this->AccessMember(totalPCMFramesCount,
        defaultTotalPCMFramesCount);
  }

  drwav_uint64& TotalPCMFramesCount()
  {
    return this->ModifyMember(totalPCMFramesCount,
        defaultTotalPCMFramesCount);
  }

  drwav_uint64 TotalSamples() const
  {
    return this->AccessMember(totalSamples, defaultTotalSamples);
  }

  drwav_uint64& TotalSamples()
  {
    return this->ModifyMember(totalSamples, defaultTotalSamples);
  }

 private:

  std::optional<drwav_uint64> audioDuration;
  std::optional<drwav_uint32> avgBytesPerSec;
  std::optional<drwav_uint16> bitPerSample;
  std::optional<drwav_uint16> channels;
  std::optional<drwav_uint64> fileBitRate;
  std::optional<drwav_uint32> sampleRate;
  std::optional<drwav_uint64> totalFramesRead;
  std::optional<drwav_uint64> totalPCMFramesCount;
  std::optional<drwav_uint64> totalSamples;

  constexpr static const drwav_uint64 defaultAudioDuration    = 0;
  constexpr static const drwav_uint32 defaultAvgBytesPerSec   = 0;
  constexpr static const drwav_uint16 defaultBitPerSample     = 0;
  constexpr static const drwav_uint16 defaultChannels         = 0;
  constexpr static const drwav_uint64 defaultFileBitRate      = 0;
  constexpr static const drwav_uint32 defaultSampleRate       = 0;
  constexpr static const drwav_uint64 defaultTotalFramesRead  = 0;
  constexpr static const drwav_uint64 defaultTotalPCMFramesCount = 0;
  constexpr static const drwav_uint64 defaultTotalSamples     = 0;
};

template<>
struct IsDataOptions<AudioOptions>
{
  constexpr static bool value = true;
};

} // namespace mlpack

#endif
