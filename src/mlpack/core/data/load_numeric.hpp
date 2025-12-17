/**
 * @file core/data/load_numeric.hpp
 * @author Omar Shrit
 *
 * Load numeric csv using Armadillo parser. Distinguish between the cases, if
 * we are loading with transpose, NaN, etc.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_NUMERIC_HPP
#define MLPACK_CORE_DATA_LOAD_NUMERIC_HPP

#include "text_options.hpp"
#include "load_categorical.hpp"
#include "load_dense.hpp"
#include "load_sparse.hpp"

namespace mlpack {
namespace data {

template<typename MatType>
bool LoadNumeric(const std::string& filename,
                 MatType& matrix,
                 std::fstream& stream,
                 TextOptions& opts)
{
  bool success = false;

  TextOptions txtOpts(std::move(opts));
  if constexpr (IsSparseMat<MatType>::value)
  {
    success = LoadSparse(filename, matrix, txtOpts, stream);
  }
  else if (txtOpts.Categorical() ||
      (txtOpts.Format() == FileType::ARFFASCII))
  {
    success = LoadCategorical(filename, matrix, txtOpts);
  }
  else if constexpr (IsCol<MatType>::value)
  {
    success = LoadDenseCol(filename, matrix, txtOpts, stream);
  }
  else if constexpr (IsRow<MatType>::value)
  {
    success = LoadDenseRow(filename, matrix, txtOpts, stream);
  }
  else if constexpr (IsDense<MatType>::value)
  {
    success = LoadDense(filename, matrix, txtOpts, stream);
  }
  else
  {
    return HandleError("data::Load(): unknown matrix-like type given!",
        txtOpts);
  }
  opts = std::move(txtOpts);
  return success;
}

template<typename MatType>
bool LoadNumericMultifile(const std::vector<std::string>& filenames,
          MatType& matrix,
          const TextOptions& opts)
{
  TextOptions copyOpts(opts);
  return Load(filenames, matrix, copyOpts);
}

template<typename MatType>
bool LoadNumericMultifile(const std::vector<std::string>& filenames,
          MatType& matrix,
          TextOptions& opts)
{
  bool success = false;
  MatType tmp;
  arma::field<std::string> firstHeaders;
  if (filenames.empty())
  {
    return HandleError("Load(): given set of filenames is empty;"
        " loading failed.", opts);
  }

  for (size_t i = 0; i < filenames.size(); ++i)
  {
    success = Load(filenames.at(i), matrix, opts);
    if (opts.HasHeaders())
    {
      if (i == 0)
        firstHeaders = opts.Headers();
      else
      {
        arma::field<std::string>& headers = opts.Headers();

        // Make sure that the headers in this file match the first file's
        // headers.
        for (size_t j = 0; j < headers.size(); ++j)
        {
          if (firstHeaders.at(j) != headers.at(j))
          {
            std::stringstream oss;
            oss << "Load(): header column " << j << " in file '"
                << filenames[j] << "' ('" << headers[j] << "') does not match"
                << " header column " << j << " in first file '"
                << filenames[0] << "' ('" << firstHeaders[j] << "'); load "
                << "failed.";
            matrix.clear();
            return HandleError(oss, opts);
          }
        }
      }
    }

    if (success)
    {
      if (i == 0)
      {
        tmp = std::move(matrix);
      }
      else
      {
        if (!opts.NoTranspose()) // if transpose
        {
          if (tmp.n_rows != matrix.n_rows)
          {
            std::stringstream oss;
            oss << "Load(): dimension mismatch; file '" << filenames[i]
                << "' has " << matrix.n_rows << " dimensions, but first file "
                << "'" << filenames[0] << "' has " << tmp.n_rows
                << " dimensions.";
            return HandleError(oss, opts);
          }
          else
            tmp = join_rows(tmp, matrix);
        }
        else
        {
          if (tmp.n_cols != matrix.n_cols)
          {
            std::stringstream oss;
            oss <<  "Load(): dimension mismatch; file '" << filenames[i]
                << "' has " << matrix.n_cols << " dimensions, but first file "
                << "'" << filenames[0] << "' has " << tmp.n_cols
                << " dimensions.";
            return HandleError(oss, opts);
          }
          else
          {
            tmp = join_cols(tmp, matrix);
          }
        }
      }
    }
    else
      break;
  }

  if (success)
    matrix = std::move(tmp);

  return success;
}

} // namespace data
} // namespace mlpack

#endif
