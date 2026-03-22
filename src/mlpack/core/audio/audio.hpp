/**
 * @file core/data/audio.hpp
 * @author Omar Shrit
 *
 * Header to include dr_mp3 and dr_wav from dr_libs in mlpack, in addition to
 * allow the user to disable all of these includes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_AUDIO_AUDIO_HPP
#define MLPACK_CORE_AUDIO_AUDIO_HPP

//
// MP3 Includes
//

#undef  DRMP3_API
#define DRMP3_API  static

#undef  DRMP3_PRIVATE
#define DRMP3_PRIVATE  static

#ifndef DR_MP3_IMPLEMENTATION
  #define DR_MP3_IMPLEMENTATION
#endif

#if defined(MLPACK_USE_SYSTEM_DR_LIBS)
  #if __has_include(<dr_mp3.h>)
    #include <dr_mp3.h>
  #else
    #pragma warning("System's dr_mp3 not found; including bundled dr_mp3")
    #include "bundled/dr_mp3.h"
#endif

#else

#include "bundled/dr_mp3.h"

#endif

//
// WAV Includes
//

#undef  DRWAV_API
#define DRWAV_API  static

#undef  DRWAV_PRIVATE
#define DRWAV_PRIVATE  static

#ifndef DR_WAV_IMPLEMENTATION
  #define DR_WAV_IMPLEMENTATION
#endif

#if defined(MLPACK_USE_SYSTEM_DR_LIBS)
  #if __has_include(<dr_wav.h>)
    #include <dr_wav.h>
  #else
    #pragma warning("System's dr_wav not found; including bundled dr_wav")
    #include "bundled/dr_wav.h"
  #endif

#else

#include "bundled/dr_wav.h"

#endif

//
// Non-template wrappers for dr_libs functions.
//
// dr_libs functions are declared static, meaning each translation unit gets
// its own private copy.  When these static functions are called directly from
// template functions (e.g. LoadWAV<eT>), the compiler may place them inside
// the same COMDAT group as the template instantiation.  If two translation
// units instantiate the same template, the linker deduplicates the COMDAT
// groups and may discard one copy of the static function while keeping
// references to it from another translation unit, causing a linker error.
//
// By wrapping every dr_libs call in a non-template inline function, the
// static dr_libs functions are only ever referenced from inline (non-template)
// code, which uses vague linkage (pick-any semantics) and avoids the COMDAT
// deduplication problem.
//

// --- dr_wav wrappers ---

inline bool DrWavInitFile(drwav* wav, const char* filename)
{
  return drwav_init_file(wav, filename, nullptr);
}

inline bool DrWavInitFileWrite(drwav* wav, const char* filename,
    const drwav_data_format* format)
{
  return drwav_init_file_write(wav, filename, format, nullptr);
}

inline void DrWavUninit(drwav* wav)
{
  drwav_uninit(wav);
}

inline size_t DrWavReadF32(drwav* wav, size_t framesToRead, float* buf)
{
  return static_cast<size_t>(
      drwav_read_pcm_frames_f32(wav, framesToRead, buf));
}

inline size_t DrWavReadS16(drwav* wav, size_t framesToRead, int16_t* buf)
{
  return static_cast<size_t>(
      drwav_read_pcm_frames_s16(wav, framesToRead, buf));
}

inline size_t DrWavReadS32(drwav* wav, size_t framesToRead, int32_t* buf)
{
  return static_cast<size_t>(
      drwav_read_pcm_frames_s32(wav, framesToRead, buf));
}

inline size_t DrWavWritePCMFrames(drwav* wav, size_t framesToWrite,
    const void* data)
{
  return static_cast<size_t>(
      drwav_write_pcm_frames(wav, framesToWrite, data));
}

// --- dr_mp3 wrappers ---

inline bool DrMp3InitFile(drmp3* mp3, const char* filename)
{
  return drmp3_init_file(mp3, filename, nullptr);
}

inline void DrMp3Uninit(drmp3* mp3)
{
  drmp3_uninit(mp3);
}

inline size_t DrMp3GetPCMFrameCount(drmp3* mp3)
{
  return static_cast<size_t>(drmp3_get_pcm_frame_count(mp3));
}

inline size_t DrMp3ReadF32(drmp3* mp3, size_t framesToRead, float* buf)
{
  return static_cast<size_t>(
      drmp3_read_pcm_frames_f32(mp3, framesToRead, buf));
}

inline size_t DrMp3ReadS16(drmp3* mp3, size_t framesToRead, int16_t* buf)
{
  return static_cast<size_t>(
      drmp3_read_pcm_frames_s16(mp3, framesToRead, buf));
}

#endif // MLPACK_CORE_AUDIO_AUDIO_HPP
