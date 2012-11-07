/***
 * @file arma_extend.hpp
 * @author Ryan Curtin
 *
 * Include Armadillo extensions which currently are not part of the main
 * Armadillo codebase.
 *
 * This will allow the use of the ccov() function (which performs the same
 * function as cov(trans(X)) but without the cost of computing trans(X)).  This
 * also gives sparse matrix support, if it is necessary.
 */
#ifndef __MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP
#define __MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP

// Use 64-bit indices (define uword as u64), but only if size_t is that size.
// Basically, this will use 64-bit indices on 64-bit systems and 32-bit indices
// on 32-bit systems (yes, there are exceptions).
#if (ULONG_MAX > 0xffffffff)
  #define ARMA_64BIT_WORD
#endif

// Add constructors for sparse vectors (these are only added if sparse support
// is enabled).
#define ARMA_EXTRA_COL_PROTO mlpack/core/arma_extend/Col_extra_bones.hpp
#define ARMA_EXTRA_COL_MEAT  mlpack/core/arma_extend/Col_extra_meat.hpp
#define ARMA_EXTRA_ROW_PROTO mlpack/core/arma_extend/Row_extra_bones.hpp
#define ARMA_EXTRA_ROW_MEAT  mlpack/core/arma_extend/Row_extra_meat.hpp

#include <armadillo>

// To get CSV support on versions of Armadillo prior to 2.0.0, we'll do this.  I
// feel dirty, but I think it's the best we can do.
#if (ARMA_VERSION_MAJOR < 2)
  #define csv_ascii (ppm_binary + 1) // ppm_binary is the last in the old enums.
#endif


namespace arma {
  // ccov()
  #include "op_ccov_proto.hpp"
  #include "op_ccov_meat.hpp"
  #include "glue_ccov_proto.hpp"
  #include "glue_ccov_meat.hpp"
  #include "fn_ccov.hpp"

  // inplace_reshape()
  #include "fn_inplace_reshape.hpp"
};

#endif
