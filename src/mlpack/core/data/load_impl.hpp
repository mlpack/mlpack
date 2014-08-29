/**
 * @file load_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templatized load() function defined in load.hpp.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_DATA_LOAD_IMPL_HPP
#define __MLPACK_CORE_DATA_LOAD_IMPL_HPP

// In case it hasn't already been included.
#include "load.hpp"

#include <algorithm>
#include <mlpack/core/util/timers.hpp>

namespace mlpack {
namespace data {

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          bool fatal,
          bool transpose)
{
  Timer::Start("loading_data");

  // First we will try to discriminate by file extension.
  size_t ext = filename.rfind('.');
  if (ext == std::string::npos)
  {
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Cannot determine type of file '" << filename << "'; "
          << "no extension is present." << std::endl;
    else
      Log::Warn << "Cannot determine type of file '" << filename << "'; "
          << "no extension is present.  Load failed." << std::endl;

    return false;
  }

  // Get the extension and force it to lowercase.
  std::string extension = filename.substr(ext + 1);
  std::transform(extension.begin(), extension.end(), extension.begin(),
      ::tolower);

  // Catch nonexistent files by opening the stream ourselves.
  std::fstream stream;
  stream.open(filename.c_str(), std::fstream::in);

  if (!stream.is_open())
  {
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Cannot open file '" << filename << "'. " << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "'; load failed."
          << std::endl;

    return false;
  }

  bool unknownType = false;
  arma::file_type loadType;
  std::string stringType;

  if (extension == "csv")
  {
    loadType = arma::csv_ascii;
    stringType = "CSV data";
  }
  else if (extension == "txt")
  {
    // This could be raw ASCII or Armadillo ASCII (ASCII with size header).
    // We'll let Armadillo do its guessing (although we have to check if it is
    // arma_ascii ourselves) and see what we come up with.

    // This is taken from load_auto_detect() in diskio_meat.hpp
    const std::string ARMA_MAT_TXT = "ARMA_MAT_TXT";
    char* rawHeader = new char[ARMA_MAT_TXT.length() + 1];
    std::streampos pos = stream.tellg();

    stream.read(rawHeader, std::streamsize(ARMA_MAT_TXT.length()));
    rawHeader[ARMA_MAT_TXT.length()] = '\0';
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (std::string(rawHeader) == ARMA_MAT_TXT)
    {
      loadType = arma::arma_ascii;
      stringType = "Armadillo ASCII formatted data";
    }
    else // It's not arma_ascii.  Now we let Armadillo guess.
    {
      loadType = arma::diskio::guess_file_type(stream);

      if (loadType == arma::raw_ascii) // Raw ASCII (space-separated).
        stringType = "raw ASCII formatted data";
      else if (loadType == arma::csv_ascii) // CSV can be .txt too.
        stringType = "CSV data";
      else // Unknown .txt... we will throw an error.
        unknownType = true;
    }

    delete[] rawHeader;
  }
  else if (extension == "bin")
  {
    // This could be raw binary or Armadillo binary (binary with header).  We
    // will check to see if it is Armadillo binary.
    const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";
    char *rawHeader = new char[ARMA_MAT_BIN.length() + 1];

    std::streampos pos = stream.tellg();

    stream.read(rawHeader, std::streamsize(ARMA_MAT_BIN.length()));
    rawHeader[ARMA_MAT_BIN.length()] = '\0';
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (std::string(rawHeader) == ARMA_MAT_BIN)
    {
      stringType = "Armadillo binary formatted data";
      loadType = arma::arma_binary;
    }
    else // We can only assume it's raw binary.
    {
      stringType = "raw binary formatted data";
      loadType = arma::raw_binary;
    }

    delete[] rawHeader;
  }
  else if (extension == "pgm")
  {
    loadType = arma::pgm_binary;
    stringType = "PGM data";
  }
  else if (extension == "h5" || extension == "hdf5" || extension == "hdf" ||
           extension == "he5")
  {
#ifdef ARMA_USE_HDF5
    loadType = arma::hdf5_binary;
    stringType = "HDF5 data";
#else
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Attempted to load '" << filename << "' as HDF5 data, but "
          << "Armadillo was compiled without HDF5 support.  Load failed."
          << std::endl;
    else
      Log::Warn << "Attempted to load '" << filename << "' as HDF5 data, but "
          << "Armadillo was compiled without HDF5 support.  Load failed."
          << std::endl;

    return false;
#endif
  }
  else // Unknown extension...
  {
    unknownType = true;
    loadType = arma::raw_binary; // Won't be used; prevent a warning.
    stringType = "";
  }

  // Provide error if we don't know the type.
  if (unknownType)
  {
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Unable to detect type of '" << filename << "'; "
          << "incorrect extension?" << std::endl;
    else
      Log::Warn << "Unable to detect type of '" << filename << "'; load failed."
          << " Incorrect extension?" << std::endl;

    return false;
  }

  // Try to load the file; but if it's raw_binary, it could be a problem.
  if (loadType == arma::raw_binary)
    Log::Warn << "Loading '" << filename << "' as " << stringType << "; "
        << "but this may not be the actual filetype!" << std::endl;
  else
    Log::Info << "Loading '" << filename << "' as " << stringType << ".  "
        << std::flush;

  const bool success = matrix.load(stream, loadType);

  if (!success)
  {
    Log::Info << std::endl;
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Loading from '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Loading from '" << filename << "' failed." << std::endl;

    return false;
  }
  else
    Log::Info << "Size is " << (transpose ? matrix.n_cols : matrix.n_rows)
        << " x " << (transpose ? matrix.n_rows : matrix.n_cols) << ".\n";

  // Now transpose the matrix, if necessary.
  if (transpose)
    matrix = trans(matrix);

  Timer::Stop("loading_data");

  // Finally, return the success indicator.
  return success;
}

}; // namespace data
}; // namespace mlpack

#endif
