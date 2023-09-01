#include "b.hpp"

// Include the static implementation of all STB functions.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef MLPACK_HAS_NO_STB_DIR
  #include <stb_image.h>
  #include <stb_image_write.h>
#else
  #include <stb/stb_image.h>
  #include <stb/stb_image_write.h>
#endif

void B::B()
{
  // Do nothing, just to check if the STB library is a working version.
}
