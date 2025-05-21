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

  return matrix.load(filename, ToArmaFileType(opts.Format()));
}

} // namespace data
} // namespace mlpack

#endif
