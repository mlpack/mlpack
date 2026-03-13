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

template<typename MatType>
bool SaveWAV(const std::string& file,
             MatType& matrix,
             AudioOptions& opts)
{
  size_t framesWritten = 0;
  // We cannot save filetype other than wav.
  if (!opts.Format() == FileType::WAV)
  {
    return HandleError("SaveWav(): Only WAV format is supported."
       " Please specify the file extension or the FileFormat.", opts);
  }

  // Convert float [-1,1] → int16
  std::vector<int16_t> pcm16(m_allSamples.size());
  for (size_t i = 0; i < m_allSamples.size(); ++i)
  {
    float clamped = std::clamp(m_allSamples[i], -1.0f, 1.0f);
    pcm16[i] = static_cast<int16_t>(clamped * 32767.0f);
  }

  drwav_data_format format;
  //  format.container     = drwav_container_riff;
  //  format.format        = DR_WAVE_FORMAT_PCM;
  format.channels      = opts.Channels();
  format.sampleRate    = opts.SampleRate();
  format.bitsPerSample = 16;

  drwav wav;
  if (!drwav_init_file_write(&wav, file, &format, nullptr))
  {
    return HandleError("SaveWav(): Failed to open wav file for writing."
       " Please check the file and try again.", opts);
  }

  // Write all frames (1 frame = 1 sample for mono)
  framesWritten = (size_t)drwav_write_pcm_frames(&wav, pcm16.size(), pcm16.data());

  if (framesWritten != pcm16.size())
  {
    return HandleError("SaveWav(): The number of written Frames mismatches "
       "the number of frames to be stored.", opts);
  }

  drwav_uninit(&wav);
  return true;
}

} //namespace mlpack

#endif
