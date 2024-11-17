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
// STB is part of mlpack. If the user does not want to use the integrated
// version of STB then please define MLPACK_NO_STB in your CMake file.
//
#define MLPACK_STB
#ifdef MLPACK_NO_STB
  #undef MLPACK_STB
#endif

#ifdef MLPACK_DISABLE_BFD_DL
  #undef MLPACK_HAS_BFD_DL
#endif

#endif
