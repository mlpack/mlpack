/***
 * @file arma_extend.hpp
 * @author Ryan Curtin
 *
 * Include Armadillo extensions which currently are not part of the main
 * Armadillo codebase.
 *
 * This will allow the use of the ccov() function (which performs the same
 * function as cov(trans(X)) but without the cost of computing trans(X)).
 */
#ifndef __ARMA_EXTEND_H
#define __ARMA_EXTEND_H

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
};

#endif
