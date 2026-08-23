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

template<typename MatType>
bool SaveMatrix(const MatType& matrix,
                TextOptions& opts,
                const std::string& filename,
                std::fstream& stream)
{
  bool success = false;
  if (opts.Format() == FileType::CSVASCII)
  {
    // Build Armadillo flags for saving based on our settings.
    // Note that MissingToNan() is ignored.
    arma::csv_opts::opts flags =
        NoTransposeOpt(opts.NoTranspose()) +
        HasHeadersOpt(opts.HasHeaders()) +
        SemicolonOpt(opts.Semicolon());

    stream.close(); // save directly to the file
    if (opts.HasHeaders())
    {
      success = matrix.save(arma::csv_name(filename, opts.Headers(), flags),
          arma::csv_ascii);
    }
    else
    {
      success = matrix.save(arma::csv_name(filename, flags), arma::csv_ascii);
    }
  }
#ifdef ARMA_USE_HDF5
  else if (opts.Format() == FileType::HDF5Binary)
  {
    if (opts.NoTranspose())
    {
      success = matrix.save(filename);
    }
    else
    {
      success = matrix.save(arma::hdf5_name(filename, "",
          arma::hdf5_opts::trans));
    }
  }
#endif
  else
  {
    // Other formats cannot be automatically transposed by Armadillo so we must
    // do it manually.
    if (!opts.NoTranspose())
    {
      MatType tmp = matrix.t();
      success = tmp.save(stream, opts.ArmaFormat());
    }
    else
    {
      success = matrix.save(stream, opts.ArmaFormat());
    }
  }

  return success;
}

} // namespace mlpack

#endif
