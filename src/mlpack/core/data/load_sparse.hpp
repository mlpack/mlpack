/**
 * @file core/data/load_sparse.hpp
 * @author Omar Shrit
 *
 * Implementation of templatized load() function defined in load.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_SPARSE_HPP
#define MLPACK_CORE_DATA_LOAD_SPARSE_HPP

#include <mlpack/prereqs.hpp>

#include "text_options.hpp"

namespace mlpack {
namespace data {

template <typename eT>
bool LoadSparse(const std::string& filename,
                arma::SpMat<eT>& matrix,
                TextOptions& opts,
                std::fstream& stream)
{
  bool success;
  // There is still a small amount of differentiation that needs to be done:
  // if we got a text type, it could be a coordinate list.  We will make an
  // educated guess based on the shape of the input.
  if (opts.Format() == FileType::RawASCII)
  {
    // Get the number of columns in the file.  If it is the right shape, we
    // will assume it is sparse.
    const size_t cols = CountCols(stream);
    if (cols == 3)
    {
      // We have the right number of columns, so assume the type is a
      // coordinate list.
      opts.Format() = FileType::CoordASCII;
    }
  }

  // Filter out invalid types.
  if ((opts.Format() == FileType::PGMBinary) ||
      (opts.Format() == FileType::PPMBinary) ||
      (opts.Format() == FileType::ArmaASCII) ||
      (opts.Format() == FileType::RawBinary))
  {
    std::stringstream oss;
    oss << "Cannot load '" << filename << "' with type "
        << opts.FileTypeToString() << " into a sparse matrix; format is "
        << "only supported for dense matrices.";
    return HandleError(oss, opts);
  }
  else if (opts.Format() == FileType::CSVASCII)
  {
    // Armadillo sparse matrices can't load CSVs, so we have to load a separate
    // matrix to do that.  If the CSV has three columns, we assume it's a
    // coordinate list.
    arma::Mat<eT> dense;
    success = dense.load(stream, opts.ArmaFormat());
    if (dense.n_cols == 3)
    {
      arma::umat locations = arma::conv_to<arma::umat>::from(
          dense.cols(0, 1).t());
      matrix = arma::SpMat<eT>(locations, dense.col(2));
    }
    else
    {
      matrix = arma::conv_to<arma::SpMat<eT>>::from(dense);
    }
  }
  else
  {
    success = matrix.load(stream, opts.ArmaFormat());
  }

  if (!opts.NoTranspose())
  {
    // It seems that there is no direct way to use inplace_trans() on
    // sparse matrices.
    matrix = matrix.t();
  }

  return success;
}

} // namespace data
} // namespace mlpack

#endif
