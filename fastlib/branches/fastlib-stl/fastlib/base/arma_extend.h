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

#include <armadillo>

namespace arma {
  // ccov()
  #include "arma_extend/op_ccov_proto.hpp"
  #include "arma_extend/op_ccov_meat.hpp"
  #include "arma_extend/glue_ccov_proto.hpp"
  #include "arma_extend/glue_ccov_meat.hpp"
  #include "arma_extend/fn_ccov.hpp"
};

#endif
