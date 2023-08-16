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

#ifndef MLPACK_HAS_BFD_DL
/**
 * 
 */
// #define MLPACK_HAS_BFD_DL
#endif

#ifndef MLPACK_HAS_STB
// #define MLPACK_HAS_STB
#endif

#ifndef MLPACK_GIT_VERSION
// #define MLPACK_GIT_VERSION
#endif

// These macros can be defined to disable support.
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
