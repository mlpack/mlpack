/**
 * @file core/data/save_audio.hpp
 * @author Omar Shrit
 *
 * Save audio data functions implementation for wav files only.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_AUDIO_HPP
#define MLPACK_CORE_DATA_SAVE_AUDIO_HPP

#include "audio_options.hpp"

namespace mlpack {

// WAV saving helper utilities:
//
// These handle actually calling dr_wav saving functions, after potentially
// mapping the input matrix to the desired save format, with the minimum number
// of copies.

/**
 * Save a matrix of floating-point data into a WAV file using the target format
 * of eT1 for each element.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalFP(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix)
{
  // No matter what, we have to clamp to the range [-1, 1] for safety, so a
  // temporary is needed.
  arma::Mat<eT1> tmp = arma::clamp(arma::conv_to<arma::Mat<eT1>>::from(matrix),
      (eT1) -1, (eT1) 1);
  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, tmp.memptr());
}

/**
 * Save a matrix to a WAV file, using the exact same format as the matrix is
 * already storing.  This expects eT1/eT2 to be an integral type.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<std::is_same_v<eT1, eT2>>* = 0)
{
  // The type is the same, so we can write directly.
  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, matrix.memptr());
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called for saving signed data, when the input type is
 * a floating-point type.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<std::is_signed_v<eT1>>* = 0)
{
  // We need to convert our floating-point numbers (which are expected to be in
  // the range [-1, 1]) to the full range of eT1s.
  arma::Mat<eT1> tmp = arma::conv_to<arma::Mat<eT1>>::from(
      arma::clamp(matrix, (eT2) -1, (eT2) 1) * std::numeric_limits<eT1>::max());
  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, tmp.memptr());
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called for saving unsigned data, when the input type
 * is a floating-point type.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<!std::is_signed_v<eT1>>* = 0)
{
  // We need to convert our floating-point numbers (which are expected to be in
  // the range [-1, 1]) to the full range of eT1s.  Since eT1 is unsigned, we do
  // this by shifting to [0, 2] and then multiplying by half the representable
  // range of eT1.
  arma::Mat<eT1> tmp = arma::conv_to<arma::Mat<eT1>>::from(
      (arma::clamp(matrix, (eT2) -1, (eT2) 1) + (eT2) 1) *
      (std::numeric_limits<eT1>::max() / 2));
  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, tmp.memptr());
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called for saving as a format whose signedness is
 * different than the given matrix's data (e.g. eT2 is uint16_t but eT1 is
 * int16_t).
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<!std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<!std::is_same_v<eT1, eT2>>* = 0,
    const typename std::enable_if_t<sizeof(eT1) == sizeof(eT2)>* = 0)
{
  // In this case we need to shift for the signedness change, but there's no
  // conversion.  Since the input is const, we unfortunately need a temporary
  // for this.
  arma::Mat<eT2> tmp(matrix);

  // Reinterpret the copy as unsigned data, if needed, and perform the shift.
  if (std::is_signed_v<eT2>)
  {
    // The output type is unsigned, so make an alias, shift, and save.
    arma::Mat<eT1> outAlias((eT1*) tmp.memptr(), tmp.n_rows, tmp.n_cols, false,
      true);
    outAlias += std::pow(2, 8 * sizeof(eT1) - 1);
    return (size_t) drwav_write_pcm_frames(&wav, totalFrames,
        outAlias.memptr());
  }
  else
  {
    // The input type is unsigned, so perform the shift, then make an alias and
    // save.
    tmp += std::pow(2, 8 * sizeof(eT2) - 1);
    arma::Mat<eT1> outAlias((eT1*) tmp.memptr(), tmp.n_rows, tmp.n_cols, false,
      true);
    return (size_t) drwav_write_pcm_frames(&wav, totalFrames,
        outAlias.memptr());
  }
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called when the signedness of eT1 and eT2 are the
 * same, but eT1 is larger, so the data needs to be expanded to fill the range.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<!std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<(sizeof(eT1) > sizeof(eT2))>* = 0,
    const typename std::enable_if_t<
        (std::is_signed_v<eT1> == std::is_signed_v<eT2>)>* = 0)
{
  // Widen the input samples, but don't perform a shift for a sign change.
  arma::Mat<eT1> tmp = arma::conv_to<arma::Mat<eT1>>::from(matrix) *
      std::pow(2, 8 * (sizeof(eT1) - sizeof(eT2)));
  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, tmp.memptr());
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called when the signedness of eT1 and eT2 are
 * different, *and* eT1 is larger.  So the data needs to be shifted *and*
 * expanded to fill the range.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<!std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<(sizeof(eT1) > sizeof(eT2))>* = 0,
    const typename std::enable_if_t<
        (std::is_signed_v<eT1> != std::is_signed_v<eT2>)>* = 0)
{
  // Perform the expansion during the conversion operation.
  arma::Mat<eT1> tmp = arma::conv_to<arma::Mat<eT1>>::from(matrix) *
      std::pow(2, 8 * (sizeof(eT1) - sizeof(eT2)));

  // The sign change must be performed on the unsigned type, so create an alias
  // if needed.
  if (std::is_signed_v<eT1>)
  {
    typedef typename std::make_unsigned_t<eT1> ueT1;
    arma::Mat<ueT1> tmpAlias((ueT1*) tmp.memptr(), tmp.n_rows, tmp.n_cols,
        false, true);
    tmpAlias -= std::pow(2, 8 * sizeof(ueT1) - 1);
  }
  else
  {
    tmp += std::pow(2, 8 * sizeof(eT1) - 1);
  }

  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, tmp.memptr());
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called when the signedness of eT1 and eT2 are the
 * same, but eT1 is smaller, so the data needs to be shrunk to not overflow the
 * range.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<!std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<(sizeof(eT1) < sizeof(eT2))>* = 0,
    const typename std::enable_if_t<
        (std::is_signed_v<eT1> == std::is_signed_v<eT2>)>* = 0)
{
  // Shrink the input samples, but don't perform a sign change shift.
  arma::Mat<eT1> tmp = arma::conv_to<arma::Mat<eT1>>::from(
      matrix / std::pow(2, 8 * (sizeof(eT2) - sizeof(eT1))));
  return (size_t) drwav_write_pcm_frames(&wav, totalFrames, tmp.memptr());
}

/**
 * Save a matrix to a WAV file using eT1 as the format to be stored in the WAV
 * file.  This overload is called when the signedness of eT1 and eT2 are
 * different, *and* eT1 is smaller.  So the data needs to be shifted *and*
 * shrunk to not overflow the range.
 */
template<typename eT1, typename eT2>
inline size_t SaveWAVInternalInt(
    drwav& wav,
    const size_t totalFrames,
    const arma::Mat<eT2>& matrix,
    const typename std::enable_if_t<!std::is_floating_point_v<eT2>>* = 0,
    const typename std::enable_if_t<(sizeof(eT1) < sizeof(eT2))>* = 0,
    const typename std::enable_if_t<
        (std::is_signed_v<eT1> != std::is_signed_v<eT2>)>* = 0)
{
  if (std::is_signed_v<eT2>)
  {
    // If eT2 is signed, then we need to use conv_to to output into a signed
    // type of the right width and then shift.  Then, we will make an unsigned
    // alias (of type eT1) and save.
    typedef typename std::make_signed_t<eT1> seT1;
    arma::Mat<seT1> tmp = arma::conv_to<arma::Mat<seT1>>::from(
        matrix / std::pow(2, 8 * (sizeof(eT2) - sizeof(eT1))));
    arma::Mat<eT1> tmpAlias((eT1*) tmp.memptr(), tmp.n_rows, tmp.n_cols, false,
        true);
    tmpAlias += std::pow(2, 8 * sizeof(eT1) - 1);

    return (size_t) drwav_write_pcm_frames(&wav, totalFrames,
        tmpAlias.memptr());
  }
  else
  {
    // If eT2 is unsigned, then we need to reinterpret the input matrix as an
    // unsigned type before conversion.  Then we can shift and save.
    typedef typename std::make_unsigned_t<eT1> ueT1;
    arma::Mat<ueT1> tmp = arma::conv_to<arma::Mat<ueT1>>::from(
        matrix / std::pow(2, 8 * (sizeof(eT2) - sizeof(eT1)))) +
        std::pow(2, 8 * sizeof(eT1) - 1);

    return (size_t) drwav_write_pcm_frames(&wav, totalFrames,
        (eT1*) tmp.memptr());
  }
}

/**
 * Save audio matrix data to a WAV file.
 *
 * Dispatches based on BitsPerSample() in AudioOptions:
 *   - 16 (or 0/unset): PCM 16-bit signed integer format.
 *     Input is clamped to [-1, 1], scaled to [-32767, 32767], and written
 *     as int16.
 *   - 32: 32-bit IEEE float format.
 *     Input is clamped to [-1, 1] and written directly as float32.
 *
 * The matrix layout must match the one produced by LoadWAV():
 *   - Shape: 1 x (totalPCMFrames * channels)
 *   - Interleaved channel samples: [L0, R0, L1, R1, ..., LN, RN]
 *
 * @param file Path to the output WAV file.
 * @param matrix Armadillo matrix containing audio samples.
 * @param opts AudioOptions with Channels() and SampleRate() set.
 * @return true on success, false on failure.
 */
template<typename eT>
bool SaveAudio(const std::string& file,
               const arma::Mat<eT>& matrix,
               AudioOptions& opts)
{
  if (opts.Format() != FileType::WAV)
  {
    return HandleError("SaveAudio(): Only WAV format is supported."
       " Please specify the file extension as `.wav` or the FileFormat as "
       "`FileType::WAV`.", opts);
  }

  if (opts.Channels() == 0 || opts.SampleRate() == 0)
  {
    std::stringstream oss;
    oss << "SaveAudio(): Number of channels or sample rate is not set. Please"
        << " set AudioOptions::Channels() or AudioOptions::SampleRate()"
        << " before saving.";
    return HandleError(oss, opts);
  }

  if (opts.BitsPerSample() != 8 && opts.BitsPerSample() != 16 &&
      opts.BitsPerSample() != 32 && opts.BitsPerSample() != 64)
  {
    if (opts.BitsPerSample() != 0 && opts.Fatal())
      Log::Fatal << "SaveAudio(): invalid BitsPerSample() value: "
          << opts.BitsPerSample() << "; must be 8/12/16/24/32/64!";
    else
      Log::Warn << "SaveAudio(): invalid BitsPerSample() value: "
          << opts.BitsPerSample() << "; using size of given data instead ("
          << (8 * sizeof(eT)) << ").";

    opts.BitsPerSample() = 8 * sizeof(eT);
  }

  opts.TotalFrames() = matrix.n_elem / opts.Channels();

  drwav_data_format dataFormat;
  dataFormat.container     = drwav_container_riff;
  dataFormat.channels      = opts.Channels();
  dataFormat.sampleRate    = opts.SampleRate();
  dataFormat.bitsPerSample = opts.BitsPerSample();

  if (opts.BitsPerSample() == 32 && std::is_floating_point_v<eT>)
    dataFormat.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  else if (opts.BitsPerSample() == 64 && std::is_floating_point_v<eT>)
    dataFormat.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  else
    dataFormat.format = DR_WAVE_FORMAT_PCM;

  drwav wav;
  if (!drwav_init_file_write(&wav, file.c_str(), &dataFormat, nullptr))
  {
    std::stringstream oss;
    oss << "SaveAudio(): failed to open WAV file '"
        << file << "'; please check the file path and permissions.";
    return HandleError(oss, opts);
  }

  size_t framesWritten = 0;
  if (dataFormat.format == DR_WAVE_FORMAT_IEEE_FLOAT)
  {
    if (opts.BitsPerSample() == 32)
    {
      framesWritten = SaveWAVInternalFP<float>(wav, opts.TotalFrames(), matrix);
    }
    else
    {
      framesWritten = SaveWAVInternalFP<double>(wav, opts.TotalFrames(),
          matrix);
    }
  }
  else
  {
    // Saving as PCM; the input is either integer or float.
    if (opts.BitsPerSample() == 8)
    {
      framesWritten = SaveWAVInternalInt<uint8_t>(wav, opts.TotalFrames(),
          matrix);
    }
    else if (opts.BitsPerSample() == 16)
    {
      framesWritten = SaveWAVInternalInt<int16_t>(wav, opts.TotalFrames(),
          matrix);
    }
    else if (opts.BitsPerSample() == 32)
    {
      framesWritten = SaveWAVInternalInt<int32_t>(wav, opts.TotalFrames(),
          matrix);
    }
    else if (opts.BitsPerSample() == 64)
    {
      framesWritten = SaveWAVInternalInt<int64_t>(wav, opts.TotalFrames(),
          matrix);
    }
  }

  drwav_uninit(&wav);

  if (framesWritten != opts.TotalFrames())
  {
    std::stringstream oss;
    oss << "SaveAudio(): Frame count mismatch: expected to write "
        << opts.TotalFrames() << " frames but only wrote "
        << framesWritten << " frames.";
    return HandleError(oss, opts);
  }

  opts.TotalSamples() = matrix.n_elem;
  opts.AudioDuration() = opts.TotalFrames() / opts.SampleRate();

  return true;
}

} //namespace mlpack

#endif
