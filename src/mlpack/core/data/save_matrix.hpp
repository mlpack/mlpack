/**
 * @file core/data/save_matrix.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Internal implementation of matrix save function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_MATRIX_HPP
#define MLPACK_CORE_DATA_SAVE_MATRIX_HPP

namespace mlpack {
namespace data {

template<typename MatType, typename DataOptionsType>
bool SaveMatrix(const MatType& matrix,
                const DataOptionsType& opts,
#ifdef ARMA_USE_HDF5
                const std::string& filename,
#else
                const std::string& /* filename */,
#endif
                std::fstream& stream)
{
  bool success = false;
  if (opts.Format() == FileType::HDF5Binary)
  {
#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    success = matrix.save(filename, opts.ArmaFormat());
#endif
  }
  else
  {
    success = matrix.save(stream, opts.ArmaFormat());
  }
  return success;
}

} // namespace data
} // namespace mlpack

#endif
