/***
 * @file arma_extend.h
 *
 * Include Armadillo extensions which currently are not part of the main
 * Armadillo codebase.
 *
 * This will allow the use of the ccov() function (which performs the same
 * function as cov(trans(X)) but without the cost of computing trans(X)).
 */
#ifndef __ARMA_EXTEND_H
#define __ARMA_EXTEND_H

// Define our own extensions.  These will be included in Cube_bones.hpp (or
// Cube_proto.hpp) and Mat_bones.hpp (or Mat_meat.hpp).
#define ARMA_EXTRA_MAT_PROTO mlpack/core/arma_extend/Mat_extra_bones.hpp
#define ARMA_EXTRA_MAT_PROTO mlpack/core/arma_extend/Mat_extra_bones.hpp

#include <armadillo>

// To get CSV support on versions of Armadillo prior to 2.0.0, we'll do this.  I
// feel dirty, but I think it's the best we can do.
#if (ARMA_VERSION_MAJOR < 2)
  #define csv_ascii (ppm_binary + 1) // ppm_binary is the last in the old enums.
#endif

namespace arma {
  // 64-bit support
  #include "typedef.hpp" // This has to come first.
  #include "traits.hpp"
  #include "promote_type.hpp"

  // ccov()
  #include "op_ccov_proto.hpp"
  #include "op_ccov_meat.hpp"
  #include "glue_ccov_proto.hpp"
  #include "glue_ccov_meat.hpp"
  #include "fn_ccov.hpp"

  // Implementation of load and save functions allowing transposes.
  #include "Mat_extra_meat.hpp"
};

#endif
