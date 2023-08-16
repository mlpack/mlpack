/**
 * @file config.hpp
 *
 * This file contains compile-time definitions that control what is supported in
 * mlpack.  In general, this file is configured by CMake, but you can edit it if
 * necessary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CONFIG_HPP
#define MLPACK_CONFIG_HPP

//
// mlpack has the capability of providing backtraces when errors are encountered
// using libbfd and libdl.  This support is only available on Linux.  If
// MLPACK_HAS_BFD_DL is enabled, then when compiling an mlpack program with
// debugging symbols (e.g. -DDEBUG), you must link with -lbfd and -ldl.
//
#ifndef MLPACK_HAS_BFD_DL
// #define MLPACK_HAS_BFD_DL
#endif

//
// mlpack provides image loading and saving support via STB, if available.  STB
// is an optional dependency of mlpack.  When STB is found on a system,
// MLPACK_HAS_STB will be defined and the files `stb_image.h` and
// `stb_image_write.h` are expected to be found in the compiler include path.
//
#ifndef MLPACK_HAS_STB
// #define MLPACK_HAS_STB
#endif

//
// If the version of mlpack is built from a git repository and is not an
// official release, then MLPACK_GIT_VERSION will be defined.  This causes
// mlpack::util::GetVersion() to return the git revision instead of the version
// number.
//
#ifndef MLPACK_GIT_VERSION
// #define MLPACK_GIT_VERSION
#endif

//
// MLPACK_COUT_STREAM is used to change the default stream for printing
// non-error messages.
//
#if !defined(MLPACK_COUT_STREAM)
 #define MLPACK_COUT_STREAM std::cout
#endif

//
// MLPACK_CERR_STREAM is used to change the stream for printing warnings and
// errors.
//
#if !defined(MLPACK_CERR_STREAM)
 #define MLPACK_CERR_STREAM std::cerr
#endif

//
// These macros can be defined to disable support that is defined above.  (This
// is useful if you cannot or do not want to modify config.hpp.)
//
#ifdef MLPACK_DISABLE_STB
  #ifdef MLPACK_HAS_STB
    #undef MLPACK_HAS_STB
  #endif
#endif

#ifdef MLPACK_DISABLE_BFD_DL
  #ifdef MLPACK_HAS_BFD_DL
    #undef MLPACK_HAS_BFD_DL
  #endif
#endif

#endif
