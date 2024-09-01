/**
 * @file core/util/omp_reductions.hpp
 * @author Mark Fischinger 
 *
 * Custom OpenMP reductions. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_OMP_REDUCTIONS_HPP
#define MLPACK_CORE_UTIL_OMP_REDUCTIONS_HPP

namespace mlpack {

// Custom reduction for arma::mat
#pragma omp declare reduction(matAdd : arma::mat : omp_out += omp_in) \
    initializer(omp_priv = arma::mat(omp_orig.n_rows, omp_orig.n_cols))

// Custom reduction for arma::Col<size_t>
#pragma omp declare reduction(colAdd : arma::Col<size_t> : omp_out += omp_in) \
    initializer(omp_priv = arma::Col<size_t>(omp_orig.n_elem))

} // namespace mlpack

#endif
