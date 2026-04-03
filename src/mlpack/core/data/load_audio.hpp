/**
 * @file core/data/load_audio.hpp
 * @author Omar Shrit
 * @author Ryan Curtin
 *
 * Load audio data functions implementation (WAV or MP3).
 * Supports loading as float32 or signed 16-bit PCM depending on the
 * element type of the destination matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_AUDIO_HPP
#define MLPACK_CORE_DATA_LOAD_AUDIO_HPP

#include "audio_options.hpp"
#include "map_integral_types.hpp"

namespace mlpack {

/**
 * ========================================
 *          Theoretical Concept.
 * ========================================
 *
 * The following text is valid for MP3 and WAV formats.
 *
 * PCM stands for Pulse Code Modulation.
 *
 *  Basically it a modulation technique that samples the analog data and
 *  create the digital samples for audio. This is a raw non-compressed data
 *  format.
 *  The amount of sampled data depends on the sampling frequency (e.g., 44.1
 *  KHz).
 *
 * Sample is a single numeric value representing the amplitude of the audio
 * waveform at one point in time, for one channel.
 *
 * Frame is a group of samples — one sample per channel — all captured at the
 * same instant in time.
 *
 * For examples:
 *
 *  Mono   (1 channel):  1 frame = 1 sample
 *  Stereo (2 channels): 1 frame = 2 samples (one for left, one for right)
 *  5.1 surround (6 channels): 1 frame = 6 samples
 *
 *  Visual representation for Stereo:
 *  Time ──────────────────────────────►
 *
 *   Frame 0          Frame 1          Frame 2
 *   ┌──────┬──────┐  ┌──────┬──────┐  ┌──────┬──────┐
 *   │  L₀  │  R₀  │  │  L₁  │  R₁  │  │  L₂  │  R₂  │
 *   └──────┴──────┘  └──────┴──────┘  └──────┴──────┘
 *
 * Channels are represented continuously when it comes to the following code.
 * Therefore, in the following PCM buffer. Each frame will be followed by the
 * next Frame in time as represented above.
 *
 * PCM[L0, R0, L1, R1, L2, R2, ..., LN, RN]
 *
 * Note that, the main difference between WAV and MP3, is that WAV can have
 * N number of channels. While MP3 is mostly capped at 2 channels.
 *
 * In real life, most of digital audio data, is 2 channels (L,R) whether this
 * for MP3 or WAV format.
 */

template<typename eT>
bool LoadAudio(const std::string file,
               arma::Mat<eT>& matrix,
               AudioOptions& opts)
{
  if (opts.Format() == FileType::WAV)
  {
    return LoadWAV(file, matrix, opts);
  }
  else if (opts.Format() == FileType::MP3)
  {
    return LoadMP3(file, matrix, opts);
  }
  else
  {
    std::stringstream oss;
    oss << "LoadAudio(): Only loading WAV and MP3 formats are supported.\n"
        << " Please check the filetype.";
    return HandleError(oss, opts);
  }
}

template<typename eT>
bool LoadWAV(const std::string& file,
             arma::Mat<eT>& matrix,
             AudioOptions& opts)
{
  size_t framesRead = 0;
  drwav wav;

  if (!drwav_init_file(&wav, file.c_str(), nullptr))
  {
    std::ostringstream oss;
    oss << "LoadWAV(): failed to read WAV file '" << file << "'; file is "
        << "corrupted or could not be opened.";
    return HandleError(oss.str(), opts);
  }

  opts.TotalFrames() = static_cast<size_t>(wav.totalPCMFrameCount);
  opts.Channels() = wav.channels;
  opts.SampleRate() = wav.sampleRate;
  opts.BitsPerSample() = 8 * sizeof(eT);

  if constexpr (std::is_floating_point_v<eT>)
  {
    arma::fmat samples(opts.TotalFrames() * opts.Channels(), 1);

    framesRead = static_cast<size_t>(drwav_read_pcm_frames_f32(
        &wav, opts.TotalFrames(), samples.memptr()));

    // 64 bits, 32 bits, 16 bits float.
    matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(samples));
  }
  else if constexpr (std::is_integral_v<eT> && sizeof(eT) >= 4)
  {
    arma::Mat<int32_t> samples(opts.TotalFrames() * opts.Channels(),
        1);

    framesRead = static_cast<size_t>(drwav_read_pcm_frames_s32(
        &wav, opts.TotalFrames(), samples.memptr()));

    MapSignedIntegralTypes(matrix, samples);
  }
  else if constexpr (std::is_integral_v<eT> && sizeof(eT) == 2)
  {
    arma::Mat<int16_t> samples(opts.TotalFrames() * opts.Channels(),
        1);

    framesRead = static_cast<size_t>(drwav_read_pcm_frames_s16(
        &wav, opts.TotalFrames(), samples.memptr()));

    MapSignedIntegralTypes(matrix, samples);
  }
  // Loading uint8_t is failing when using s16 function. (Not supported by
  // dr_wav) It is better to load as f32 and then map the results back to
  // uint8_t.
  else if constexpr (std::is_integral_v<eT> && sizeof(eT) == 1)
  {
    arma::fmat samples(opts.TotalFrames() * opts.Channels(), 1);
    framesRead = static_cast<size_t>(drwav_read_pcm_frames_f32(
        &wav, opts.TotalFrames(), samples.memptr()));

    if constexpr (std::is_same_v<eT, uint8_t>)
    {
      matrix = arma::conv_to<arma::Mat<uint8_t>>::from(
        (samples + 1.0f) * 127.5f);
    }
    else
    {
      matrix = arma::conv_to<arma::Mat<int8_t>>::from(samples * 127.0f);
    }
  }

  drwav_uninit(&wav);

  if (framesRead != opts.TotalFrames())
  {
    std::stringstream oss;
    oss << "LoadWAV(): Frame count mismatch: " << opts.TotalFrames()
        << "(queried) != " << framesRead <<" (read)";
    return HandleError(oss, opts);
  }

  opts.AudioDuration() = opts.TotalFrames() / opts.SampleRate();
  opts.TotalSamples() = opts.TotalFrames() * opts.Channels();

  return true;
}

template<typename eT>
bool LoadMP3(const std::string& file,
             arma::Mat<eT>& matrix,
             AudioOptions& opts)
{
  drmp3 mp3;
  size_t framesRead = 0;

  if (!drmp3_init_file(&mp3, file.c_str(), nullptr))
  {
    std::ostringstream oss;
    oss << "LoadMP3(): failed to read MP3 file '" << file << "'; file is "
        << "corrupted or could not be opened.";
    return HandleError(oss.str(), opts);
  }

  opts.TotalFrames() = static_cast<size_t>(drmp3_get_pcm_frame_count(&mp3));
  opts.Channels() = mp3.channels;
  opts.SampleRate() = mp3.sampleRate;
  opts.BitsPerSample() = 8 * sizeof(eT);

  if constexpr (std::is_floating_point_v<eT>)
  {
    arma::fmat samples(opts.TotalFrames() * opts.Channels(), 1);

    framesRead = static_cast<size_t>(drmp3_read_pcm_frames_f32(
        &mp3, opts.TotalFrames(), samples.memptr()));

    matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(samples));
  }
  else if constexpr (std::is_integral_v<eT>)
  {
    arma::Mat<int16_t> samples(opts.TotalFrames() * opts.Channels(), 1);

    framesRead = static_cast<size_t>(drmp3_read_pcm_frames_s16(
        &mp3, opts.TotalFrames(), samples.memptr()));

    MapSignedIntegralTypes(matrix, samples);
  }

  drmp3_uninit(&mp3);

  if (framesRead != opts.TotalFrames())
  {
    std::stringstream oss;
    oss << "LoadMP3(): Frame count mismatch: " << opts.TotalFrames()
        << "(queried) != " << framesRead <<" (read)";
    return HandleError(oss, opts);
  }

  opts.AudioDuration() = opts.TotalFrames() / opts.SampleRate();
  opts.TotalSamples() = opts.TotalFrames() * opts.Channels();

  return true;
}

} //namespace mlpack

#endif
