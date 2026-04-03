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
#include "map_integral_types.hpp"

namespace mlpack {

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
  size_t framesWritten = 0;

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
          << opts.BitsPerSample() << "; must be 8/16/32/64!";
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
  else if (opts.BitsPerSample() == 64 && std::is_integral_v<eT>)
    dataFormat.format = DR_WAVE_FORMAT_PCM;
  else if (opts.BitsPerSample() == 32 && std::is_integral_v<eT>)
    dataFormat.format = DR_WAVE_FORMAT_PCM;
  else if (opts.BitsPerSample() == 16)
    dataFormat.format = DR_WAVE_FORMAT_PCM;
  else if (opts.BitsPerSample() == 8)
    dataFormat.format = DR_WAVE_FORMAT_PCM;

  drwav wav;
  if (!drwav_init_file_write(&wav, file.c_str(), &dataFormat, nullptr))
  {
    std::stringstream oss;
    oss << "SaveAudio(): failed to open WAV file '"
        << file << "'; please check the file path and permissions.";
    return HandleError(oss, opts);
  }

  // Cover double and float
  if constexpr (std::is_floating_point_v<eT>)
  {
    if (opts.BitsPerSample() == 8)
    {
      arma::fmat pcm32 = arma::conv_to<arma::fmat>::from(matrix);
      pcm32.clamp(-1.0f, 1.0f);
      pcm32 = (pcm32 + 1.0f) * 127.5f;

      arma::Mat<uint8_t> pcm8 = arma::conv_to<arma::Mat<uint8_t>>::from(pcm32);
      framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), pcm8.memptr()));
    }
    else if (opts.BitsPerSample() == 16)
    {
      // We assume that the original values are in range of [-1, +1]
      arma::fmat pcm32 = arma::conv_to<arma::fmat>::from(matrix);
      pcm32.clamp(-1.0f, 1.0f);
      pcm32 *= 32767.0f;

      arma::Mat<int16_t> pcm16 = arma::conv_to<arma::Mat<int16_t>>::from(pcm32);
      framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), pcm16.memptr()));
    }
    else if (opts.BitsPerSample() == 32)
    {
      arma::fmat pcm32 = arma::conv_to<arma::fmat>::from(matrix);
      framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
        opts.TotalFrames(), pcm32.memptr()));
    }
    else if (opts.BitsPerSample() == 64)
    {
      arma::mat pcm64 = arma::conv_to<arma::mat>::from(matrix);
      framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
        opts.TotalFrames(), pcm64.memptr()));
    }
  }
  else if constexpr (std::is_integral_v<eT>)
  {
    // Create a copy as we need to modify it
    arma::Mat<eT> tmpMatrix = matrix;

    // User might have different tmpMatrix type compared to bit per sample.
    if (opts.BitsPerSample() != 8 * sizeof(eT))
    {
      // We must convert int8_t to uint8_t to match the WAV standard.
      // Handles int8_t, uint8_t
      if (opts.BitsPerSample() == 8)
      {
        arma::Mat<uint8_t> pcm;
        if constexpr (std::is_signed_v<eT>)
        {
          MapSignedIntegralTypes(pcm, tmpMatrix);
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), pcm.memptr()));
        }
        else if constexpr (!std::is_signed_v<eT>)
        {
          tmpMatrix /= std::pow(2, 8 * (sizeof(eT) - 1));
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), tmpMatrix.memptr()));
        }
      }
      // Handles int16, int32, int64
      else if constexpr (std::is_signed_v<eT> && !std::is_same_v<eT, int8_t>)
      {
        framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
              opts.TotalFrames(), tmpMatrix.memptr()));
      }
      // Handles uint16, uint32, uint64
      else if constexpr (!std::is_signed_v<eT> && !std::is_same_v<eT, uint8_t>)
      {
        if (opts.BitsPerSample() == 16)
        {
          arma::Mat<int16_t> pcm;
          MapUnsignedIntegralTypes(pcm, tmpMatrix);
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
                opts.TotalFrames(), pcm.memptr()));
        }
        else if (opts.BitsPerSample() == 32)
        {
          arma::Mat<int32_t> pcm;
          MapUnsignedIntegralTypes(pcm, tmpMatrix);
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
                opts.TotalFrames(), pcm.memptr()));
        }
        else if (opts.BitsPerSample() == 64)
        {
          arma::Mat<int64_t> pcm;
          MapUnsignedIntegralTypes(pcm, tmpMatrix);
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
                opts.TotalFrames(), pcm.memptr()));
        }
      }
    }
    else // bitsPerSample == sizeof(eT), or user did not specify bitPerSample
    {
      // We must convert int8_t to uint8_t to match the WAV standard.
      if (std::is_same_v<eT, int8_t>)
      {
        arma::Mat<uint8_t> pcm;
        MapSignedIntegralTypes(pcm, tmpMatrix);
        framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), pcm.memptr()));
      }
      else if constexpr (std::is_same_v<eT, uint8_t>)
      {
        framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), tmpMatrix.memptr()));
      }
      else
      {
        // Handles: int16_t, int32_t, int64_t
        if constexpr (std::is_signed_v<eT>)
        {
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
            opts.TotalFrames(), tmpMatrix.memptr()));
        }
        // Handles: uint16_t, uint32_t, uint64_t
        else if constexpr (!std::is_signed_v<eT>)
        {
          typedef std::make_signed_t<eT> seT;
          arma::Mat<seT> pcm;
          MapUnsignedIntegralTypes(pcm, tmpMatrix);
          framesWritten = static_cast<size_t>(drwav_write_pcm_frames(&wav,
              opts.TotalFrames(), pcm.memptr()));
        }
      }
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
