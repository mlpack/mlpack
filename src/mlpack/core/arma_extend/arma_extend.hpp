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

// Add batch constructor for sparse matrix (if version <= 3.810.0).
#define ARMA_EXTRA_SPMAT_PROTO mlpack/core/arma_extend/SpMat_extra_bones.hpp
#define ARMA_EXTRA_SPMAT_MEAT  mlpack/core/arma_extend/SpMat_extra_meat.hpp

#include <armadillo>

namespace arma {
  // u64/s64
  #include "typedef.hpp"
  #include "traits.hpp"
  #include "promote_type.hpp"
  #include "restrictors.hpp"
  #include "hdf5_misc.hpp"

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
