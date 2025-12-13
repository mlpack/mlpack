/**
 * @file core/data/save_numeric.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Internal implementation of numeric save function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_NUMERIC_HPP
#define MLPACK_CORE_DATA_SAVE_NUMERIC_HPP

#include "text_options.hpp"
#include "save_sparse.hpp"
#include "save_dense.hpp"

namespace mlpack {
namespace data {

template<typename ObjectType, typename DataOptionsType>
bool SaveNumeric(const std::string& filename,
                 const ObjectType& matrix,
                 std::fstream& stream,
                 DataOptionsBase<DataOptionsType>& opts)
{
  bool success = false;

  TextOptions txtOpts(std::move(opts));
  if constexpr (IsSparseMat<ObjectType>::value)
  {
    success = SaveSparse(matrix, txtOpts, filename, stream);
  }
  else if constexpr (IsCol<ObjectType>::value)
  {
    const bool oldNoTranspose = txtOpts.NoTranspose();
    txtOpts.NoTranspose() = true; // Force no transpose for a column.
    success = SaveDense(matrix, txtOpts, filename, stream);
    txtOpts.NoTranspose() = oldNoTranspose;
  }
  else if constexpr (IsRow<ObjectType>::value)
  {
    const bool oldNoTranspose = txtOpts.NoTranspose();
    txtOpts.NoTranspose() = false; // Force transpose for a row.
    success = SaveDense(matrix, txtOpts, filename, stream);
    txtOpts.NoTranspose() = oldNoTranspose;
  }
  else if constexpr (IsDense<ObjectType>::value)
  {
    success = SaveDense(matrix, txtOpts, filename, stream);
  }
  opts = std::move(txtOpts);

  return success;
}

} // namespace data
} // namespace mlpack

#endif
