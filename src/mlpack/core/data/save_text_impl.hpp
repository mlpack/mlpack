/**
 * @file save_text_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of Save() for text formats using TextOptions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_TEXT_IMPL_HPP 
#define MLPACK_CORE_DATA_SAVE_TEXT_IMPL_HPP

// In case it hasn't already been included.
#include "save.hpp"

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

// Save a Sparse Matrix
template<typename eT>
bool SaveSparse(const arma::SpMat<eT>& matrix,
                TextOptions& opts,
                const std::string& filename,
                std::fstream& stream)
{
  bool success = false;
  arma::SpMat<eT> tmp;

  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    arma::SpMat<eT> tmp = trans(matrix);
    success = SaveMatrix(tmp, opts, filename, stream);
  }
  else
    success = SaveMatrix(matrix, opts, filename, stream);

  return success;
}

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
    opts.NoTranspose() = true;
    success = SaveDense(matrix, txtOpts, filename, stream);
  }
  else if constexpr (IsRow<ObjectType>::value)
  {
    opts.NoTranspose() = false;
    success = SaveDense(matrix, txtOpts, filename, stream);
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
