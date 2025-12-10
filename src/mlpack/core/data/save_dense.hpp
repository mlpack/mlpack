/**
 * @file core/data/save_dense.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Internal implementation of dense matrix save function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_DENSE_HPP
#define MLPACK_CORE_DATA_SAVE_DENSE_HPP

#include "save_matrix.hpp"

namespace mlpack {
namespace data {

template<typename eT>
bool SaveDense(const arma::Mat<eT>& matrix,
               TextOptions& opts,
               const std::string& filename,
               std::fstream& stream)
{
  bool success = false;
  arma::Mat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = trans(matrix);
    success = SaveMatrix(tmp, opts, filename, stream);
  }
  else
    success = SaveMatrix(matrix, opts, filename, stream);

  return success;
}

} // namespace data
} // namespace mlpack

#endif
