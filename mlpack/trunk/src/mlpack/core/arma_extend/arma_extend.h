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

namespace arma {
  #include "typedef.hpp" // This has to come first.
}

#include <armadillo>

namespace arma {
  // 64-bit support
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
