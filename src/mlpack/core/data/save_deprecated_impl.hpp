/**
 * @file core/data/save_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of save functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_DEPRECATED_IMPL_HPP
#define MLPACK_CORE_DATA_SAVE_DEPRECATED_IMPL_HPP

// In case it hasn't already been included.
#include "save_deprecated.hpp"
#include "extension.hpp"

namespace mlpack {
namespace data {

/*
 * btw, the following two functions are not documented anywhere
 * If they are not exposed to the public API then I can delete them
 * If they are exposed to the public API, then they are deprecated.
 * @rcurtin please comment:
 */
template<typename eT>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal,
          FileType inputSaveType)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false, inputSaveType);
}

template<typename eT>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
bool Save(const std::string& filename,
          const arma::Row<eT>& rowvec,
          const bool fatal,
          FileType inputSaveType)
{
  return Save(filename, rowvec, fatal, true, inputSaveType);
}

// Save a Sparse Matrix
template<typename eT>
bool Save(const std::string& filename,
          const arma::SpMat<eT>& matrix,
          const bool fatal,
          bool transpose)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;

  return Save(filename, matrix, opts);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          const bool fatal,
          bool transpose,
          FileType inputSaveType)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.FileFormat() = inputSaveType;

  return Save(filename, matrix, opts);
}

//! Save a model to file.
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f)
{
  ModelOptions opts;
  opts.ObjectName() = name;
  opts.Fatal() = fatal;
  opts.DataFormat() = f;

  return Save(filename, t, opts);
}



} // namespace data
} // namespace mlpack

#endif
