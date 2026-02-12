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

#ifndef DR_MP3_IMPLEMENTATION
  #define DR_MP3_IMPLEMENTATION
#endif

#ifndef DR_WAV_IMPLEMENTATION
  #define DR_WAV_IMPLEMENTATION
#endif

#if defined(MLPACK_USE_SYSTEM_MP3)
  #if __has_include(<dr_mp3.h>)
    #include <dr_mp3.h>
  #else
    #pragma warning("System's dr_mp3 not found; including bundled dr_mp3")
    #include "bundled/dr_mp3.h"
#endif

#else

#include "bundled/dr_mp3.h"

#endif

#if defined(MLPACK_USE_SYSTEM_WAV)
  #if __has_include(<dr_wav.h>)
    #include <dr_wav.h>
  #else
    #pragma warning("System's dr_wav not found; including bundled dr_wav")
    #include "bundled/dr_wav.h"
#endif

#else

#include "bundled/dr_wav.h"

#endif

#endif // MLPACK_CODE_AUDIO_AUDIO_HPP
