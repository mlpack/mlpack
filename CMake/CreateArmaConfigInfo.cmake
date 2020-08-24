# Using the CMake tools to create the file arma_config.hpp, which contains
# information on the Armadillo configuration when mlpack was compiled.  This
# assumes ${ARMADILLO_INCLUDE_DIR} is set.  In addition, we must be careful to
# avoid overwriting arma_config.hpp with the exact same information, because
# this may trigger a new complete rebuild, which is undesired.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/arma_config.hpp")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/arma_config.hpp"
      OLD_FILE_CONTENTS)
else()
  set(OLD_FILE_CONTENTS "")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(ARMA_HAS_64BIT_WORD 0)
else()
  set(ARMA_HAS_64BIT_WORD 1)
endif()

# Now use the value we gathered to generate the new file contents.
if(ARMA_HAS_64BIT_WORD EQUAL 0)
  set(ARMA_64BIT_WORD_DEFINE "#define MLPACK_ARMA_NO64BIT_WORD")
else()
  set(ARMA_64BIT_WORD_DEFINE "#define MLPACK_ARMA_64BIT_WORD")
endif()

# Next we need to know if we are compiling with OpenMP support.
# Other places in the CMake configuration should have already done the
# find(OpenMP).
if (OPENMP_FOUND)
  set(ARMA_HAS_OPENMP_DEFINE "#define MLPACK_ARMA_USE_OPENMP")
else ()
  set(ARMA_HAS_OPENMP_DEFINE "#define MLPACK_ARMA_DONT_USE_OPENMP")
endif ()

set(NEW_FILE_CONTENTS
"/**
 * @file arma_config.hpp
 *
 * This is an autogenerated file which contains the configuration of Armadillo
 * at the time mlpack was built.  If you modify anything in here by hand, your
 * warranty is void, your house may catch fire, and we're not going to call the
 * police when your program segfaults so hard that robbers come to your house
 * and take everything you own.  If you do decide, against better judgment, to
 * modify anything at all in this file, and you are reporting a bug, be
 * absolutely certain to mention that you've done something stupid in this file
 * first.
 *
 * In short: don't touch this file.
 */
#ifndef MLPACK_CORE_UTIL_ARMA_CONFIG_HPP
#define MLPACK_CORE_UTIL_ARMA_CONFIG_HPP

${ARMA_64BIT_WORD_DEFINE}

${ARMA_HAS_OPENMP_DEFINE}

#endif
")

# Did the contents of the file change at all?  If not, don't write it.
if(NOT "${OLD_FILE_CONTENTS}" STREQUAL "${NEW_FILE_CONTENTS}")
  # We have a reason to write the new file.
  message(STATUS "Regenerating arma_config.hpp.")
  file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/arma_config.hpp")
  file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/arma_config.hpp"
      "${NEW_FILE_CONTENTS}")
endif()

