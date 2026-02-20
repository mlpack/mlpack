/**
 * @file core/data/httplib.hpp
 * @author Omar Shrit
 *
 * Header to include cpp-httplib in mlpack, in addition to allow the user to
 * disable all of these includes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_HTTPLIB_HTTPLIB_HPP
#define MLPACK_CORE_HTTPLIB_HTTPLIB_HPP

#ifdef MLPACK_USE_HTTPS
  #define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#if defined(MLPACK_USE_SYSTEM_HTTPLIB)
#if defined __has_include
  #if __has_include(<httplib.h>)
    #include <httplib.h>
  #else
    #pragma warning("System's httplib not found; including bundled httplib")
    #include "bundled/httplib.h"
  #endif
#endif

#else

#ifdef MLPACK_ENABLE_HTTPLIB
  #ifndef MLPACK_DISABLE_HTTPLIB
    // Now include httplib headers
    #pragma message("httplib has been included")
    #include "bundled/httplib.h"
  #endif
#endif

#endif

#endif
