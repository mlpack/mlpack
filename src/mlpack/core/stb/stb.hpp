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


#if defined(MLPACK_USE_SYSTEM_STB)

#include <mlpack/prereqs.hpp>

#ifndef STB_IMAGE_STATIC
  #define STB_IMAGE_STATIC
#endif

#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
  #define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_IMPLEMENTATION
  #define STB_IMAGE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
  #define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#if defined __has_include
  #if __has_include("stb_image_resize.h")
    #include "stb_image_resize.h"
  #elif __has_include("stb/stb_image_resize.h")
    #include "stb/stb_image_resize.h"
  #else
    #define MLPACK_DISABLE_STB
    #pragma message("Warning: STB disabled; stb_image_resize.h header not found")
  #endif
#endif

#if defined __has_include
  #if __has_include("stb_image.h")
    #include "stb_image.h"
  #elif __has_include("stb/stb_image.h")
    #include "stb/stb_image.h"
  #else
    #define MLPACK_DISABLE_STB
    #pragma message("Warning: STB disabled; stb_image.h header not found")
  #endif
#endif

#if defined __has_include
  #if __has_include("stb_image_write.h")
    #include "stb_image_write.h"
  #elif __has_include("stb/stb_image_write.h")
    #include "stb/stb_image_write.h"
  #else
    #define MLPACK_DISABLE_STB
    #pragma message("Warning: STB disabled; stb_image_write.h header not found")
  #endif
#endif

#elif defined(MLPACK_STB)

#include <mlpack/prereqs.hpp>

#ifndef STB_IMAGE_STATIC
  #define STB_IMAGE_STATIC
#endif

#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
  #define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_IMPLEMENTATION
  #define STB_IMAGE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
  #define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

// Now include STB headers
#include "stb_image_resize.h"
#include "stb_image.h"
#include "stb_image_write.h"

#endif

#endif
