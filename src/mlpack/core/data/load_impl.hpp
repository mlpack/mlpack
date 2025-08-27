/**
 * @file core/data/load_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 * @author Gopi Tatiraju
 *
 * Implementation of templatized load() function defined in load.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_IMPL_HPP

// In case it hasn't already been included.
#include "load.hpp"

#include <algorithm>
#include <exception>

#include "extension.hpp"
#include "string_algorithms.hpp"

namespace mlpack {
namespace data {

// The following functions are kept for backward compatibility,
// Please remove them when we release mlpack 5.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Format() = inputLoadType;

  return Load(filename, matrix, opts);
}

// For loading data into sparse matrix
template <typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Format() = inputLoadType;

  return Load(filename, matrix, opts);
}

// For loading data into a column vector
template <typename eT>
bool Load(const std::string& filename,
          arma::Col<eT>& vec,
          const bool fatal)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  return Load(filename, vec, opts);
}

// For loading data into a raw vector
template <typename eT>
bool Load(const std::string& filename,
          arma::Row<eT>& rowvec,
          const bool fatal)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  return Load(filename, rowvec, opts);
}

// Load with mappings.  Unfortunately we have to implement this ourselves.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          DatasetInfo& info,
          const bool fatal,
          const bool transpose)
{
  TextOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Categorical() = true;
  opts.DatasetInfo() = info;

  bool success = Load(filename, matrix, opts);

  info = opts.DatasetInfo();

  return success;
}

template<typename MatType, typename DataOptionsType>
bool Load(const std::string& filename,
          MatType& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  DataOptionsType tmpOpts(opts);
  return Load(filename, matrix, tmpOpts);
}

template<typename MatType>
bool LoadMatrix(const std::string& filename,
                MatType& matrix,
                std::fstream& stream,
                TextOptions& txtOpts)
{
  bool success = false;
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
    success = LoadCol(filename, matrix, txtOpts, stream);
  }
  else if constexpr (IsRow<MatType>::value)
  {
    success = LoadRow(filename, matrix, txtOpts, stream);
  }
  else if constexpr (IsDense<MatType>::value)
  {
    success = LoadDense(filename, matrix, txtOpts, stream);
  }
  else
  {
    if (txtOpts.Fatal())
      Log::Fatal << "data::Load(): unknown matrix-like type given!"
          << std::endl;
    else
      Log::Warn << "data::Load(): unknown matrix-like type given!"
          << std::endl;

    return false;
  }
  return success;
}

template<typename ObjectType, typename DataOptionsType>
bool Load(const std::string& filename,
          ObjectType& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  Timer::Start("loading_data");
  static_assert(!IsArma<ObjectType>::value || !IsSparseMat<ObjectType>::value
      || !HasSerialize<ObjectType>::value, "mlpack can load Armadillo"
      " matrices or serialized mlpack models only; please use a known type.");
  std::fstream stream;
  bool success = OpenFile(filename, opts, true, stream);
  if (!success)
  {
    Timer::Stop("loading_data");
    return false;
  }

  success = DetectFileType<ObjectType>(filename, opts, true, &stream);
  if (!success)
  {
    Timer::Stop("loading_data");
    return false;
  }

  if constexpr (IsArma<ObjectType>::value || IsSparseMat<ObjectType>::value)
  {
    TextOptions txtOpts(std::move(opts));
    success = LoadMatrix(filename, matrix, stream, txtOpts);
    opts = std::move(txtOpts);
  }
  else if constexpr (HasSerialize<ObjectType>::value)
  {
    success = LoadModel(matrix, opts, stream);
  }
  else
  {
    if (opts.Fatal())
      Log::Fatal << "DataOptionsType is unknown!  Please use a known type "
          << "or provide specific overloads." << std::endl;
    else
      Log::Warn << "DataOptionsType is unknown!  Please use a known type "
          << "or provide specific overloads." << std::endl;
    return false;
  }

  if (!success)
  {
    Timer::Stop("loading_data");
    if (opts.Fatal())
      Log::Fatal << "Loading from '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Loading from '" << filename << "' failed." << std::endl;

    return false;
  }
  else
  {
    if constexpr (IsArma<ObjectType>::value)
    {
      Log::Info << "Size is " << matrix.n_rows << " x "
          << matrix.n_cols << ".\n";
    }
  }

  Timer::Stop("loading_data");

  return success;
}

template<typename MatType>
bool LoadDense(const std::string& filename,
               MatType& matrix,
               TextOptions& opts,
               std::fstream& stream)
{
  bool success;
  if (opts.Format() != FileType::RawBinary)
    Log::Info << "Loading '" << filename << "' as "
        << opts.FileTypeToString() << ".  " << std::flush;

  // We can't use the stream if the type is HDF5.
  if (opts.Format() == FileType::HDF5Binary)
  {
    success = LoadHDF5(filename, matrix, opts);
  }
  else if (opts.Format() == FileType::CSVASCII)
  {
    success = LoadCSVASCII(filename, matrix, opts);

    if (matrix.col(0).is_zero())
      Log::Warn << "data::Load(): the first line in '" << filename << "' was "
          << "loaded as all zeros; if the first row is headers, specify "
          << "`HasHeaders() = true` in the given DataOptions." << std::endl;
  }
  else
  {
    if (opts.Format() == FileType::RawBinary)
      Log::Warn << "Loading '" << filename << "' as "
        << opts.FileTypeToString() << "; "
        << "but this may not be the actual filetype!" << std::endl;

    success = matrix.load(stream, opts.ArmaFormat());
    if (!opts.NoTranspose())
      inplace_trans(matrix);
  }
  return success;
}

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
    if (opts.Fatal())
      Log::Fatal << "Cannot load '" << filename << "' with type "
          << opts.FileTypeToString() << " into a sparse matrix; format is "
          << "only supported for dense matrices." << std::endl;
    else
      Log::Warn << "Cannot load '" << filename << "' with type "
          << opts.FileTypeToString() << " into a sparse matrix; format is "
          << "only supported for dense matrices; load failed." << std::endl;

    return false;
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

template<typename eT>
bool LoadCategorical(const std::string& filename,
                     arma::Mat<eT>& matrix,
                     TextOptions& opts)
{
  // Get the extension and load as necessary.
  Timer::Start("loading_data");

  // Get the extension.
  std::string extension = Extension(filename);
  bool success = false;

  if (extension == "csv" || extension == "tsv" || extension == "txt")
  {
    Log::Info << "Loading '" << filename << "' as CSV dataset.  " << std::flush;
    LoadCSV loader(filename, opts.Fatal());
    success = loader.LoadCategoricalCSV(matrix, opts);
    if (!success)
    {
      Timer::Stop("loading_data");
      return false;
    }
  }
  else if (extension == "arff")
  {
    Log::Info << "Loading '" << filename << "' as ARFF dataset.  "
        << std::flush;
    success = LoadARFF(filename, matrix, opts.DatasetInfo(), opts.Fatal());
    if (!success)
    {
      Timer::Stop("loading_data");
      return false;
    }
    // Retranspose back as we are transposing by default
    if (opts.NoTranspose())
    {
      inplace_trans(matrix);
    }
  }
  else
  {
    // The type is unknown.
    Timer::Stop("loading_data");
    if (opts.Fatal())
      Log::Fatal << "Unable to detect type of '" << filename << "'; "
          << "Incorrect extension?" << std::endl;
    else
      Log::Warn << "Unable to detect type of '" << filename << "'; load failed."
          << " Incorrect extension?" << std::endl;

    return false;
  }

  Log::Info << "Size is " << matrix.n_rows << " x " << matrix.n_cols << ".\n";

  Timer::Stop("loading_data");

  return true;
}

template<typename Object>
bool LoadModel(Object& objectToSerialize,
               DataOptionsBase<PlainDataOptions>& opts,
               std::fstream& stream)
{
  try
  {
    if (opts.Format() == FileType::XML)
    {
      cereal::XMLInputArchive ar(stream);
      ar(cereal::make_nvp("model", objectToSerialize));
    }
    else if (opts.Format() == FileType::JSON)
    {
     cereal::JSONInputArchive ar(stream);
     ar(cereal::make_nvp("model", objectToSerialize));
    }
    else if (opts.Format() == FileType::BIN)
    {
      cereal::BinaryInputArchive ar(stream);
      ar(cereal::make_nvp("model", objectToSerialize));
    }

    return true;
  }
  catch (cereal::Exception& e)
  {
    if (opts.Fatal())
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

template<typename MatType>
bool Load(const std::vector<std::string>& filenames,
          MatType& matrix,
          const TextOptions& opts)
{
  TextOptions copyOpts(opts);
  return Load(filenames, matrix, copyOpts);
}

template<typename MatType>
bool Load(const std::vector<std::string>& filenames,
          MatType& matrix,
          TextOptions& opts)
{
  bool success = false;
  MatType tmp;
  arma::field<std::string> firstHeaders;
  if (filenames.empty())
  {
    if (opts.Fatal())
    {
      Log::Fatal << "Load(): given set of filenames is empty; loading failed."
          << std::endl;
    }
    else
    {
      Log::Warn << "Load(): given set of filenames is empty; loading failed."
          << std::endl;
      return false;
    }
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
            if (opts.Fatal())
            {
              Log::Fatal << "Load(): header column " << j << " in file '"
                  << filenames[j] << "' ('" << headers[j] << "') does not match"
                  << " header column " << j << " in first file '"
                  << filenames[0] << "' ('" << firstHeaders[j] << "'); load "
                  << "failed." << std::endl;
            }
            else
            {
              Log::Warn << "Load(): header column " << j << " in file '"
                  << filenames[j] << "' ('" << headers[j] << "') does not match"
                  << " header column " << j << " in first file '"
                  << filenames[0] << "' ('" << firstHeaders[j] << "'); load "
                  << "failed." << std::endl;
              matrix.clear();
              return false;
            }
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
            if (opts.Fatal())
            {
              Log::Fatal << "Load(): dimension mismatch; file '" << filenames[i]
                  << "' has " << matrix.n_rows << " dimensions, but first file "
                  << "'" << filenames[0] << "' has " << tmp.n_rows
                  << " dimensions." << std::endl;
            }
            else
            {
              Log::Warn << "Load(): dimension mismatch; file '" << filenames[i]
                  << "' has " << matrix.n_rows << " dimensions, but first file "
                  << "'" << filenames[0] << "' has " << tmp.n_rows
                  << " dimensions." << std::endl;
              return false;
            }
          }
          else
            tmp = join_rows(tmp, matrix);
        }
        else
        {
          if (tmp.n_cols != matrix.n_cols)
          {
            if (opts.Fatal())
            {
              Log::Fatal << "Load(): dimension mismatch; file '" << filenames[i]
                  << "' has " << matrix.n_cols << " dimensions, but first file "
                  << "'" << filenames[0] << "' has " << tmp.n_cols
                  << " dimensions." << std::endl;
            }
            else
            {
              Log::Warn << "Load(): dimension mismatch; file '" << filenames[i]
                  << "' has " << matrix.n_cols << " dimensions, but first file "
                  << "'" << filenames[0] << "' has " << tmp.n_cols
                  << " dimensions." << std::endl;
              return false;
            }
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
