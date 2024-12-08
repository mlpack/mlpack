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

#include <mlpack/prereqs.hpp>

#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#endif

#ifndef STB_IMAGE_STATIC
#define STB_IMAGE_STATIC
#endif

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

#endif
