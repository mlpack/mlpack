/**
 * @file core/data/load.hpp
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

namespace mlpack {
namespace data {

template<typename MatType>
bool LoadCSVASCII(const std::string& filename,
                  MatType& matrix,
                  const TextOptions& opts)
{
  bool success = false;

// show a warning if strict is availabe on not.
#if ARMA_VERSION_MAJOR < 12
  #pragma warning ("MissingToNan() support requires minimum Armadillo version >= 12.0.")
  #define NOT_MISSING_TO_NAN
#endif

  if (!opts.NoTranspose() && opts.HasHeaders() &&
      !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::trans), arma::csv_ascii);

  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::trans),
        arma::csv_ascii);

  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(filename, arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers()),
        arma::csv_ascii);

  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::semicolon),
        arma::csv_ascii);

  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(filename, arma::csv_ascii);
#else
  success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict),
        arma::csv_ascii);
#endif

  else if (!opts.NoTranspose() && opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon + arma::csv_opts::trans), arma::csv_ascii);

  else if (!opts.NoTranspose() && opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(arma::csv_name(filename, opts.Headers(),
        arma::csv_opts::semicolon + arma::csv_opts::trans),
        arma::csv_ascii);
#else
    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon + arma::csv_opts::trans +
          arma::csv_opts::strict), arma::csv_ascii);
#endif

  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::trans),
        arma::csv_ascii);
#else
    success = matrix.load(arma::csv_name(filename,
          arma::csv_opts::trans + arma::csv_opts::strict), arma::csv_ascii);

#endif
   
  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename,
          arma::csv_opts::trans + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon), arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(arma::csv_name(filename, opts.Headers()),
        arma::csv_ascii);
#else
    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::strict), arma::csv_ascii);
#endif
    
  else if (opts.NoTranspose() && opts.HasHeaders() &&
            opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(arma::csv_name(filename, opts.Headers(),
        arma::csv_opts::semicolon), arma::csv_ascii);
#else
    success = matrix.load(arma::csv_name(filename, opts.Headers(),
        arma::csv_opts::strict + arma::csv_opts::semicolon), arma::csv_ascii);
#endif

  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::trans +
        arma::csv_opts::semicolon), arma::csv_ascii);
#else
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict +
          arma::csv_opts::trans + arma::csv_opts::semicolon), arma::csv_ascii);
#endif

  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

#ifdef NOT_MISSING_TO_NAN
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::semicolon),
        arma::csv_ascii);
#else
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict +
          arma::csv_opts::semicolon), arma::csv_ascii);
#endif

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
  
  return matrix.load(filename, ToArmaFileType(opts.FileFormat()));
}

} // namespace data
} // namespace mlpack

#endif
