/**
 * @file core/data/stb.hpp
 * @author Omar Shrit
 *
 * Header to include stb in mlpack, in addition to allow the user to disable
 * all of these includes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_STB_STB_HPP
#define MLPACK_CORE_STB_STB_HPP

#ifndef STB_IMAGE_STATIC
  #define STB_IMAGE_STATIC
#endif

#ifndef STB_IMAGE_WRITE_STATIC
  #define STB_IMAGE_WRITE_STATIC
#endif

#ifndef STB_IMAGE_RESIZE_STATIC
  #define STB_IMAGE_RESIZE_STATIC
#endif

#ifndef STB_IMAGE_IMPLEMENTATION
  #define STB_IMAGE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
  #define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
  #define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif

#if defined(MLPACK_USE_SYSTEM_STB)

#if defined __has_include
  #if __has_include(<stb_image.h>)
    #include <stb_image.h>
  #elif __has_include(<stb/stb_image.h>)
    #include <stb/stb_image.h>
  #else
    #pragma warning("System's STB not found; including bundled STB")
    #include "bundled/stb_image.h"
  #endif
#endif

#if defined __has_include
  #if __has_include(<stb_image_write.h>)
    #include <stb_image_write.h>
  #elif __has_include(<stb/stb_image_write.h>)
    #include <stb/stb_image_write.h>
  #else
    #pragma warning("System's STB not found; including bundled STB")
    #include "bundled/stb_image_write.h"
  #endif
#endif

#if defined __has_include
  #if __has_include(<stb_image_resize2.h>)
    #include <stb_image_resize2.h>
  #elif __has_include(<stb/stb_image_resize2.h>)
    #include <stb/stb_image_resize2.h>
  #else
    #pragma warning("System's STB not found; including bundled STB")
    #include "bundled/stb_image_resize2.h"
  #endif
#endif

#else

// Now include STB headers
#include "bundled/stb_image.h"
#include "bundled/stb_image_write.h"
#include "bundled/stb_image_resize2.h"

#endif

#endif
