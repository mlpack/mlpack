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

// Figure out if we have sparse matrix support yet.
#include <armadillo_bits/config.hpp>

#ifndef ARMA_HAS_SPMAT
  // Include forward declarations of SpMat, SpRow, and SpCol.
  namespace arma {
    #include "sparse/forward_bones.hpp"
  };

  // Now set the extra things to be included for regular Mat, Row, and Col.
  #define ARMA_EXTRA_MAT_PROTO mlpack/core/arma_extend/sparse/Mat_extra_bones.hpp
  #define ARMA_EXTRA_MAT_MEAT  mlpack/core/arma_extend/sparse/Mat_extra_meat.hpp
#endif

#include <armadillo>

// To get CSV support on versions of Armadillo prior to 2.0.0, we'll do this.  I
// feel dirty, but I think it's the best we can do.
#if (ARMA_VERSION_MAJOR < 2)
  #define csv_ascii (ppm_binary + 1) // ppm_binary is the last in the old enums.
#endif

// Include all the things we might need for sparse matrix support.
#ifndef ARMA_HAS_SPMAT

namespace arma {
  #include "sparse/traits.hpp"

  #include "sparse/Proxy.hpp"

  #include "sparse/SpValProxy_bones.hpp"
  #include "sparse/SpMat_bones.hpp"
  #include "sparse/SpCol_bones.hpp"
  #include "sparse/SpRow_bones.hpp"
  #include "sparse/SpSubview_bones.hpp"

  #include "sparse/arma_ostream_bones.hpp"

  #include "sparse/restrictors.hpp"

  #include "sparse/fn_accu.hpp"
  #include "sparse/fn_eye.hpp"
  #include "sparse/fn_ones.hpp"
//  #include "sparse/fn_qr.hpp"
  #include "sparse/fn_randn.hpp"
  #include "sparse/fn_randu.hpp"
  #include "sparse/fn_zeros.hpp"
  #include "sparse/fn_min.hpp"
  #include "sparse/fn_max.hpp"

  #include "sparse/SpValProxy_meat.hpp"
  #include "sparse/SpMat_meat.hpp"
  #include "sparse/SpCol_meat.hpp"
  #include "sparse/SpRow_meat.hpp"
  #include "sparse/SpSubview_meat.hpp"

  #include "sparse/arma_ostream_meat.hpp"
//  #include "sparse/op_dot_meat.hpp"
};

#endif

namespace arma {
  // u64
  #include "typedef.hpp"
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
