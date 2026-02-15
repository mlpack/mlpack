/**
 * @file core/data/load_wav.hpp
 * @author Omar Shrit
 *
 * Load wav data functions implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_WAV_HPP
#define MLPACK_CORE_DATA_LOAD_WAV_HPP

#include "audio_options.hpp"

namespace mlpack {

/**
 * ========================================
 *          Theorectical Concept.
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
 * Channels are represented continsously when it comes to the following code.
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
template<typename MatType>
bool LoadWav(const std::string& filename,
             MatType& matrix,
             AudioOptions& opts)
{
  drwav wav;
  drwav_uint64 framesRead;
  drwav_uint64 totalFramesRead = 0;
  arma::fmat fullFileFrames;

  if (!drwav_init_file(&wav, filename.c_str(), NULL))
  {
    return HandleError("Failed to read wav file. Please check the file "
        "and try again.", opts);
  }

  opts.TotalPCMFramesCount() = wav.totalPCMFrameCount;
  opts.Channels() = wav.channels;

  drwav_seek_to_pcm_frame(&wav, opts.TotalPCMFramesCount() / 2);
  drwav_seek_to_pcm_frame(&wav, 0);

  // The size of the array is defined by drlibs for easy memory management.
  float pcm[4096];

  std::vector<float> fullFrames;
  fullFrames.reserve(opts.TotalPCMFramesCount() * opts.Channels());
  // We will read iterately the PCM frames, each time we fill the buffer we
  // insert it to std::vector<> and repeat again until we read the entire file.
  for (;;)
  {
    framesRead = drwav_read_pcm_frames_f32(&wav,
        sizeof(pcm)/sizeof(pcm[0])/opts.Channels(), pcm);
    if (framesRead == 0)
      break;

    size_t samplesRead = framesRead * opts.Channels();
    fullFrames.insert(fullFrames.end(), pcm, pcm + samplesRead);

    opts.TotalFramesRead() += framesRead;
  }

  if (opts.TotalFramesRead() != opts.TotalPCMFramesCount())
  {
    std::stringstream oss;
    oss << "Frame count mismatch: " << (int)opts.TotalPCMFramesCount()
        << "(queried) != " << (int)opts.TotalFramesRead() <<" (read)";
    return HandleError(oss, opts);
  }

  opts.SampleRate() = wav.sampleRate;
  opts.BitsPerSample() = wav.bitsPerSample;
  opts.ContainerType() = wav.container;
  opts.AudioDuration() = opts.TotalPCMFramesCount() / opts.SampleRate();
  opts.TotalSamples() = opts.TotalPCMFramesCount() * opts.Channels();
  opts.FileBitRate() = opts.BitsPerSample() * opts.TotalSamples()
      * opts.Channels();

  matrix = arma::conv_to<arma::Mat<float>>::from(std::move(fullFrames));
  return true;
}

} //namespace mlpack

#endif
