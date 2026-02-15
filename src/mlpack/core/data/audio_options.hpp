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
    blockAlign          = other.blockAlign;
    channels            = other.channels;
    containerType       = other.containerType;
    dataChunkSize       = other.dataChunkSize;
    fileBitRate         = other.fileBitRate;
    formatTag           = other.formatTag;
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
    blockAlign          = std::move(other.blockAlign);
    channels            = std::move(other.channels);
    containerType       = std::move(other.containerType);
    dataChunkSize       = std::move(other.dataChunkSize);
    fileBitRate         = std::move(other.fileBitRate);
    formatTag           = std::move(other.formatTag);
    sampleRate          = std::move(other.sampleRate);
    totalFramesRead     = std::move(other.totalFramesRead);
    totalPCMFramesCount = std::move(other.totalPCMFramesCount);
    totalSamples        = std::move(other.totalSamples);

    // Move base members.
    DataOptionsBase<AudioOptions>::operator=(std::move(other));

    return *this;
  }

  static const char* DataDescription() { return "audio data"; }

  void Reset()
  {
    audioDuration.reset();
    avgBytesPerSec.reset();
    bitPerSample.reset();
    blockAlign.reset();
    channels.reset();
    containerType.reset();
    dataChunkSize.reset();
    fileBitRate.reset();
    formatTag.reset();
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

  drwav_uint16 BlockAlign() const
  {
    return this->AccessMember(blockAlign, defaultBlockAlign);
  }

  drwav_uint16& BlockAlign()
  {
    return this->ModifyMember(blockAlign, defaultBlockAlign);
  }

  drwav_uint16 Channels() const
  {
    return this->AccessMember(channels, defaultChannels);
  }

  drwav_uint16& Channels()
  {
    return this->ModifyMember(channels, defaultChannels);
  }

  drwav_uint64 DataChunkSize() const
  {
    return this->AccessMember(dataChunkSize, defaultDataChunkSize);
  }

  drwav_uint64& DataChunkSize()
  {
    return this->ModifyMember(dataChunkSize, defaultDataChunkSize);
  }

  drwav_uint64 FileBitRate() const
  {
    return this->AccessMember(fileBitRate, defaultFileBitRate);
  }

  drwav_uint64& FileBitRate()
  {
    return this->ModifyMember(fileBitRate, defaultFileBitRate);
  }

  drwav_uint16 FormatTag() const
  {
    return this->AccessMember(formatTag, defaultFormatTag);
  }

  drwav_uint16& FormatTag()
  {
    return this->ModifyMember(formatTag, defaultFormatTag);
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

  // we will need to create our own enum that wrapper the one from WAV
  // Also we need to be sure that the type is not MP3 when these are being
  // accessed
  // Maybe assign -1 my default ? but these are unsigned int.
  // So how we can make it easy?
//  opts.ContainerType() = wav.container;

 private:

  std::optional<drwav_uint64> audioDuration;
  std::optional<drwav_uint32> avgBytesPerSec;
  std::optional<drwav_uint16> bitPerSample;
  std::optional<drwav_uint16> blockAlign;
  std::optional<drwav_uint16> channels;
  std::optional<drwav_container> containerType;
  std::optional<drwav_uint64> dataChunkSize;
  std::optional<drwav_uint64> fileBitRate;
  std::optional<drwav_uint16> formatTag;
  std::optional<drwav_uint32> sampleRate;
  std::optional<drwav_uint64> totalFramesRead;
  std::optional<drwav_uint64> totalPCMFramesCount;
  std::optional<drwav_uint64> totalSamples;

  constexpr static const drwav_uint64 defaultAudioDuration    = 0;
  constexpr static const drwav_uint32 defaultAvgBytesPerSec   = 0;
  constexpr static const drwav_uint16 defaultBitPerSample     = 0;
  constexpr static const drwav_uint16 defaultBlockAlign       = 0;
  constexpr static const drwav_uint16 defaultChannels         = 0;
  constexpr static const drwav_container defaultContainerType = 0; // Check
  constexpr static const drwav_uint64 defaultDataChunkSize    = 0;
  constexpr static const drwav_uint64 defaultFileBitRate      = 0;
  constexpr static const drwav_uint16 defaultFormatTag        = 0;
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
