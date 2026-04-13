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

// WAV loading helper utilities:
//
// These handle actually calling dr_wav loading functions, and mapping the
// loaded input to the desired type with the minimum number of copies.

/**
 * Load a WAV file into a float32 matrix.
 */
inline size_t LoadWAVInternalFP(drwav& wav, arma::Mat<float>& matrix)
{
  // No copy or conversion is needed here: we can read directly into the
  // float32 elements of the matrix.
  return (size_t) drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount,
      matrix.memptr());
};

/**
 * Load a WAV file into a matrix with a floating-point type that's not `float`.
 */
template<typename eT>
inline size_t LoadWAVInternalFP(
    drwav& wav,
    arma::Mat<eT>& matrix,
    const typename std::enable_if_t<std::is_floating_point_v<eT>>* = 0)
{
  // A type conversion from a float32 matrix is needed, since that's all dr_wav
  // can load for us.
  arma::Mat<float> tmp(matrix.n_rows, matrix.n_cols, arma::fill::none);
  const size_t framesRead = (size_t) drwav_read_pcm_frames_f32(&wav,
      wav.totalPCMFrameCount, tmp.memptr());

  // No scaling is needed; frames are already in the range [-1.0, 1.0].
  matrix = arma::conv_to<arma::Mat<eT>>::from(tmp);
  return framesRead;
}

/**
 * Load a WAV file containing floating-point data, and then convert it to an
 * integral type.
 */
template<typename eT>
inline size_t LoadWAVInternalFP(
    drwav& wav,
    arma::Mat<eT>& matrix,
    const typename std::enable_if_t<!std::is_floating_point_v<eT>>* = 0)
{
  // Create a temporary float32 matrix to load into.
  arma::Mat<float> tmp(matrix.n_rows, matrix.n_cols, arma::fill::none);
  const size_t framesRead = (size_t) drwav_read_pcm_frames_f32(&wav,
      wav.totalPCMFrameCount, tmp.memptr());

  // Convert to the integral type after expanding to fill the range.
  if (!std::is_signed_v<eT>)
  {
    // The output type is unsigned, so shift from [-1, 1] to [0, 2], then
    // multiply to fill the whole range.
    matrix = arma::conv_to<arma::Mat<eT>>::from((tmp + 1.0) *
        (std::numeric_limits<eT>::max() / 2));
  }
  else
  {
    // The output type is signed, so multiply to fill the whole range.
    matrix = arma::conv_to<arma::Mat<eT>>::from(tmp *
        std::numeric_limits<eT>::max());
  }

  return framesRead;
}

/**
 * Given an input matrix of integral type eT2 and a desired integral output
 * type eT1 where the signedness of the types is the same (e.g. both types are
 * signed, or both types are unsigned), map the range of values in the src
 * matrix (which fill the full range of eT2s) to the full range of eT1 values.
 */
template<typename eT1, typename eT2>
inline void ConvertType(
    arma::Mat<eT1>& dest,
    arma::Mat<eT2>& src,
    const typename std::enable_if_t<
        std::is_signed_v<eT1> == std::is_signed_v<eT2>>* = 0)
{
  // Now do the mapping, if needed.
  if (sizeof(eT1) < sizeof(eT2))
  {
    // If eT1 is smaller, then we have to shrink before we convert.
    dest = arma::conv_to<arma::Mat<eT1>>::from(
        src / std::pow(2, 8 * (sizeof(eT2) - sizeof(eT1))));
  }
  else if (sizeof(eT1) > sizeof(eT2))
  {
    // If eT1 is larger, then we have to convert before we expand the range.
    dest = arma::conv_to<arma::Mat<eT1>>::from(src) *
        std::pow(2, 8 * (sizeof(eT1) - sizeof(eT2)));
  }
}

/**
 * Given a matrix of integral type eT2 and a desired integral output type eT1
 * where the signedness of the types is the same (e.g. one is unsigned and one
 * is signed), shift and map the range such that the values in the destination
 * matrix fill the whole range of eT1s.
 */
template<typename eT1, typename eT2>
inline void ConvertType(
    arma::Mat<eT1>& dest,
    arma::Mat<eT2>& src,
    const typename std::enable_if_t<
        std::is_signed_v<eT1> != std::is_signed_v<eT2>>* = 0)
{
  // First map input values to the correct range.
  if (std::is_signed_v<eT2>)
  {
    // Signed arithmetic overflow is not guaranteed by the standard, so we need
    // to make an alias.
    typedef typename std::make_unsigned_t<eT2> ueT2;
    arma::Mat<ueT2> srcAlias((ueT2*) src.memptr(), src.n_rows, src.n_cols,
        false, true);
    srcAlias += std::pow(2, 8 * sizeof(ueT2) - 1);
    // Now do the actual conversion with signedness being the same.
    ConvertType(dest, srcAlias);
  }
  else
  {
    // eT2 is unsigned so we can do the sign change directly.
    src -= std::pow(2, 8 * sizeof(eT2) - 1);
    typedef typename std::make_signed_t<eT2> seT2;
    arma::Mat<seT2> srcAlias((seT2*) src.memptr(), src.n_rows, src.n_cols,
        false, true);
    // Now do the actual conversion with signedness being the same.
    ConvertType(dest, srcAlias);
  }
}

/**
 * Load a WAV file containing integral PCM frames into a matrix whose type is
 * also integral, but a different size than the on-disk size.
 */
template<typename eT1, typename eT2>
inline size_t LoadWAVInternalInt(
    drwav& wav,
    arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<sizeof(eT1) != sizeof(eT2)>* = 0)
{
  // We have to load into a temporary matrix, and then convert.
  arma::Mat<eT1> tmp(matrix.n_rows, matrix.n_cols, arma::fill::none);
  const size_t framesRead = (size_t) drwav_read_pcm_frames(&wav,
      wav.totalPCMFrameCount, (eT1*) tmp.memptr());

  // Now convert to the right type and perform any signed/unsigned conversions
  // if needed.
  ConvertType(matrix, tmp);
  return framesRead;
}

template<typename eT1, typename eT2>
inline size_t LoadWAVInternalInt(
    drwav& wav,
    arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<sizeof(eT1) == sizeof(eT2)>* = 0,
    const typename std::enable_if_t<!std::is_same_v<eT1, eT2>>* = 0)
{
  const size_t framesRead = (size_t) drwav_read_pcm_frames(&wav,
      wav.totalPCMFrameCount, (eT1*) matrix.memptr());
  if (std::is_signed_v<eT2>)
  {
    // We loaded into a signed type, but need to shift it to unsigned.
    // However, because signed overflow is not guaranteed by the C++ standard,
    // we have to do the shifting on the unsigned version.
    matrix += std::pow(2, 8 * sizeof(eT1) - 1);
  }
  else
  {
    // We loaded into an unsigned type, but need to shift it to signed.
    // In this case, so that we are depending on unsigned overflow behavior, we
    // have to make an alias of the matrix as an unsigned matrix.
    arma::Mat<eT1> alias((eT1*) matrix.memptr(), matrix.n_rows, matrix.n_cols,
        false, true);
    alias -= std::pow(2, 8 * sizeof(eT1) - 1);
  }

  return framesRead;
}

/**
 * Load a WAV file containing integral PCM frames into a matrix whose type is
 * exactly the same as the underlying samples in the file.
 */
template<typename eT1, typename eT2>
inline size_t LoadWAVInternalInt(
    drwav& wav,
    arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<std::is_same_v<eT1, eT2>>* = 0)
{
  // In this case we can load directly into the output.
  return (size_t) drwav_read_pcm_frames(&wav, wav.totalPCMFrameCount,
      matrix.memptr());
}

/**
 * Load a WAV file containing any bitrate into a specific eT1 type (must be
 * either int16_t or int32_t!), and then convert to the matrix type eT2,
 * shifting if necessary to fill the whole range of possible eT2 values.
 */
template<typename eT1, typename eT2>
inline size_t LoadWAVInternalForce(
    drwav& wav,
    arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<sizeof(eT1) == sizeof(eT2)>* = 0)
{
  // We don't need to load into a temporary matrix, but we may need to shift the
  // results after loading, if eT2 is an unsigned type.
  size_t framesRead = 0;
  if (wav.bitsPerSample <= 16)
  {
    framesRead = (size_t) drwav_read_pcm_frames_s16(&wav,
        wav.totalPCMFrameCount, (int16_t*) matrix.memptr());
  }
  else
  {
    framesRead = (size_t) drwav_read_pcm_frames_s32(&wav,
        wav.totalPCMFrameCount, (int32_t*) matrix.memptr());
  }

  // If the destination type is unsigned, then we have to map from the signed
  // range (which we loaded) to the unsigned range.
  if (!std::is_signed_v<eT2>)
    matrix += std::pow(2, 8 * sizeof(eT2) - 1);

  return framesRead;
}

template<typename eT1, typename eT2>
inline size_t LoadWAVInternalForce(
    drwav& wav,
    arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<sizeof(eT1) != sizeof(eT2)>* = 0)
{
  // Load into a temporary matrix.  We use the _s16 and _s32 types here to force
  // whatever underlying data format we have in the WAV file into the ranges of
  // int16_t or int32_t (depending on the type of eT1).
  arma::Mat<eT1> tmp(matrix.n_rows, matrix.n_cols, arma::fill::none);
  size_t framesRead = 0;
  if (wav.bitsPerSample <= 16)
  {
    framesRead = (size_t) drwav_read_pcm_frames_s16(&wav,
        wav.totalPCMFrameCount, (int16_t*) tmp.memptr());
  }
  else
  {
    framesRead = (size_t) drwav_read_pcm_frames_s32(&wav,
        wav.totalPCMFrameCount, (int32_t*) tmp.memptr());
  }

  // Now convert the type we loaded to the desired output type, shifting the
  // range from the signed range to unsigned range if necessary.
  ConvertType(matrix, tmp);

  return framesRead;
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
  opts.BitsPerSample() = wav.bitsPerSample;

  matrix.set_size(opts.TotalFrames() * opts.Channels(), 1);

  /**
   * Depending on the type we want to load into, and the type that we detected,
   * we might need to use different loading functions provided by dr_wav.
   *
   * When possible, we use the "plain" drwav_read_pcm_frames(), and only use the
   * drwav_read_pcm_frames_f32(), drwav_read_pcm_frames_s16(), or
   * drwav_read_pcm_frames_s32() variants when we absolutely need to.
   */
  if (std::is_floating_point_v<eT> || wav.fmt.formatTag != DR_WAVE_FORMAT_PCM)
  {
    // Load into a floating-point matrix, and optionally convert to eT if
    // needed.
    framesRead = LoadWAVInternalFP(wav, matrix);
  }
  else
  {
    // When we are loading an integral type, we must load into the type
    // specified as the first template argument.  When possible (e.g. when the
    // type width is the same), the LoadWAVInternal() function will load
    // directly into the output matrix; otherwise, a temporary will be created
    // and mapped to the correct output range.
    //
    // When the bits per sample is not a typical value (8/16/32/64), then we
    // must use dr_wav's functionality to load specifically into a int16/int32
    // type, and then do any mapping after that.
    if (wav.bitsPerSample == 8)
      framesRead = LoadWAVInternalInt<uint8_t>(wav, matrix);
    else if (wav.bitsPerSample == 16)
      framesRead = LoadWAVInternalInt<int16_t>(wav, matrix);
    else if (wav.bitsPerSample == 32)
      framesRead = LoadWAVInternalInt<int32_t>(wav, matrix);
    else if (wav.bitsPerSample == 64)
      framesRead = LoadWAVInternalInt<int64_t>(wav, matrix);
    else if (wav.bitsPerSample <= 16)
      framesRead = LoadWAVInternalForce<int16_t>(wav, matrix);
    else if (wav.bitsPerSample > 16)
      framesRead = LoadWAVInternalForce<int32_t>(wav, matrix);
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

  matrix.set_size(opts.TotalFrames() * opts.Channels(), 1);

  if constexpr (std::is_floating_point_v<eT>)
  {
    if (std::is_same_v<eT, float>)
    {
      framesRead = static_cast<size_t>(drmp3_read_pcm_frames_f32(
          &mp3, opts.TotalFrames(), (float*) matrix.memptr()));
    }
    else
    {
      arma::fmat samples(opts.TotalFrames() * opts.Channels(), 1);
      framesRead = static_cast<size_t>(drmp3_read_pcm_frames_f32(
          &mp3, opts.TotalFrames(), samples.memptr()));
      matrix = arma::conv_to<arma::Mat<eT>>::from(std::move(samples));
    }
  }
  else if constexpr (std::is_integral_v<eT>)
  {
    if (std::is_same_v<eT, int16_t>)
    {
      framesRead = static_cast<size_t>(drmp3_read_pcm_frames_s16(
          &mp3, opts.TotalFrames(), (int16_t*) matrix.memptr()));
    }
    else
    {
      arma::Mat<int16_t> samples(opts.TotalFrames() * opts.Channels(), 1);
      framesRead = static_cast<size_t>(drmp3_read_pcm_frames_s16(
          &mp3, opts.TotalFrames(), samples.memptr()));
      ConvertType(matrix, samples);
    }
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
