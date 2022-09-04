/***
 * @file core/arma_extend/arma_extend.hpp
 * @author Ryan Curtin
 *
 * Include Armadillo extensions which currently are not part of the main
 * Armadillo codebase.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP
#define MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP

// Add vec_type, col_type, and row_type to Mat and SpMat.
// TODO: refactor and remove these!
#define ARMA_EXTRA_MAT_PROTO mlpack/core/arma_extend/Mat_extra_bones.hpp
#define ARMA_EXTRA_SPMAT_PROTO mlpack/core/arma_extend/SpMat_extra_bones.hpp

#include <armadillo>

#endif
