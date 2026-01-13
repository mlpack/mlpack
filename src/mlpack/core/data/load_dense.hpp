/**
 * @file core/data/load_dense.hpp
 * @author Omar Shrit
 *
 * Load an Armadillo matrix from file.  This is necessary because Armadillo does
 * not transpose matrices on input, and it allows us to give better error
 * output.
 * Load numeric csv using Armadillo parser. Distinguish between the cases, if
 * we are loading with transpose, NaN, etc.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_DENSE_HPP
#define MLPACK_CORE_DATA_LOAD_DENSE_HPP

#include "text_options.hpp"

namespace mlpack {
namespace data {

// These help assemble the Armadillo csv_opts flags.
inline const arma::csv_opts::opts& NoTransposeOpt(const bool noTranspose)
{
  if (noTranspose)
    return arma::csv_opts::none;
  else
    return arma::csv_opts::trans;
}

inline const arma::csv_opts::opts& HasHeadersOpt(const bool hasHeaders)
{
  if (hasHeaders)
    return arma::csv_opts::with_header;
  else
    return arma::csv_opts::no_header;
}

inline const arma::csv_opts::opts& SemicolonOpt(const bool semicolon)
{
  if (semicolon)
    return arma::csv_opts::semicolon;
  else
    return arma::csv_opts::none;
}

inline const arma::csv_opts::opts& MissingToNanOpt(const bool missingToNan)
{
  if (missingToNan)
  {
    #if ARMA_VERSION_MAJOR >= 12
    return arma::csv_opts::strict;
    #else
    return arma::csv_opts::none;
    #endif
  }
  else
  {
    return arma::csv_opts::none;
  }
}

template<typename MatType>
bool LoadCSVASCII(const std::string& filename,
                  MatType& matrix,
                  TextOptions& opts)
{
  bool success = false;

  // Show a warning if strict is not available.
  #if ARMA_VERSION_MAJOR < 12
  if (opts.MissingToNan())
  {
    if (opts.Fatal())
    {
      Log::Fatal << "data::Load(): MissingToNan() requires Armadillo version "
          << ">= 12.0!" << std::endl;
    }
    else
    {
      Log::Warn << "data::Load(): MissingToNan() requires Armadillo version "
          << ">= 12.0!  Option ignored." << std::endl;
    }
  }
  #endif

  // Build Armadillo flags for loading based on our settings.
  arma::csv_opts::opts flags =
      NoTransposeOpt(opts.NoTranspose()) +
      HasHeadersOpt(opts.HasHeaders()) +
      SemicolonOpt(opts.Semicolon()) +
      MissingToNanOpt(opts.MissingToNan());

  if (opts.HasHeaders())
  {
    success = matrix.load(arma::csv_name(filename, opts.Headers(), flags),
        arma::csv_ascii);
  }
  else
  {
    success = matrix.load(arma::csv_name(filename, flags), arma::csv_ascii);
  }

  return success;
}

template<typename eT, typename DataOptionsType>
bool LoadHDF5(const std::string& filename,
              arma::Mat<eT>& matrix,
              const DataOptionsBase<DataOptionsType>& opts)
{
#ifndef ARMA_USE_HDF5
  // Ensure that HDF5 is supported.
  Timer::Stop("loading_data");
  if (opts.Fatal())
    Log::Fatal << "Attempted to load '" << filename << "' as HDF5 data, but "
        << "Armadillo was compiled without HDF5 support.  Load failed."
        << std::endl;
  else
    Log::Warn << "Attempted to load '" << filename << "' as HDF5 data, but "
        << "Armadillo was compiled without HDF5 support.  Load failed."
        << std::endl;

  return false;
#endif

  return matrix.load(filename, opts.ArmaFormat());
}

// Load column vector.
template<typename eT>
bool LoadDenseCol(const std::string& filename,
                  arma::Col<eT>& vec,
                  TextOptions& opts,
                  std::fstream& stream)
{
  // First load into auxiliary matrix.
  arma::Mat<eT> tmp;
  opts.NoTranspose() = true; // false Transpose
  bool success = LoadDense(filename, tmp, opts, stream);
  if (!success)
  {
    vec.clear();
    return false;
  }

  // Now check the size to see that it is a vector, and return a vector.
  if (tmp.n_cols > 1)
  {
    if (tmp.n_rows > 1)
    {
      std::stringstream oss;
      oss << "Matrix in file '" << filename << "' is not a vector, but"
            << " instead has size " << tmp.n_rows << "x" << tmp.n_cols << "!";
      vec.clear();
      return HandleError(oss, opts);
    }
    else
    {
      /**
       * It's loaded as a row vector (more than one column).  So we need to
       * manually modify the shape of the matrix.  We can do this without
       * damaging the data since it is only a vector.
       */
      arma::access::rw(tmp.n_rows) = tmp.n_cols;
      arma::access::rw(tmp.n_cols) = 1;

      /**
       * Now we can call the move operator, but it has to be the move operator
       * for Mat, not for Col.  This will avoid copying the data.
       */
      *((arma::Mat<eT>*) &vec) = std::move(tmp);
      return true;
    }
  }
  else
  {
    // It's loaded as a column vector.  We can call the move constructor
    // directly.
    *((arma::Mat<eT>*) &vec) = std::move(tmp);
    return true;
  }
}

// Load row vector.
template<typename eT>
bool LoadDenseRow(const std::string& filename,
                  arma::Row<eT>& rowvec,
                  TextOptions& opts,
                  std::fstream& stream)
{
  arma::Mat<eT> tmp;
  opts.NoTranspose() = true; // false Transpose
  bool success = LoadDense(filename, tmp, opts, stream);
  if (!success)
  {
    rowvec.clear();
    return false;
  }

  if (tmp.n_rows > 1)
  {
    if (tmp.n_cols > 1)
    {
      std::stringstream oss;
      oss << "Matrix in file '" << filename << "' is not a vector, but"
          << " instead has size " << tmp.n_rows << "x" << tmp.n_cols << "!";
      rowvec.clear();
      return HandleError(oss, opts);
    }
    else
    {
      /**
       * It's loaded as a column vector (more than one row).  So we need to
       * manually modify the shape of the matrix.  We can do this without
       * damaging the data since it is only a vector.
       */
      arma::access::rw(tmp.n_cols) = tmp.n_rows;
      arma::access::rw(tmp.n_rows) = 1;

      /**
       * Now we can call the move operator, but it has to be the move operator
       * for Mat, not for Col.  This will avoid copying the data.
       */
      *((arma::Mat<eT>*) &rowvec) = std::move(tmp);
      return true;
    }
  }
  else
  {
    // It's loaded as a row vector.  We can call the move constructor directly.
    *((arma::Mat<eT>*) &rowvec) = std::move(tmp);
    return true;
  }
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

} // namespace data
} // namespace mlpack

#endif
