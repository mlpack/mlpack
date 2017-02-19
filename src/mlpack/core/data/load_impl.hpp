/**
 * @file load_impl.hpp
 * @author Ryan Curtin
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
#include "load_csv.hpp"
#include "extension.hpp"

#include <exception>
#include <algorithm>
#include <mlpack/core/util/timers.hpp>

#include "load_csv.hpp"
#include "load.hpp"
#include "extension.hpp"

#include <boost/algorithm/string/trim.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include "load_arff.hpp"

namespace mlpack {
namespace data {

namespace details{

template<typename Tokenizer>
std::vector<std::string> ToTokens(Tokenizer &lineTok)
{
  std::vector<std::string> tokens;
  std::transform(std::begin(lineTok), std::end(lineTok),
                 std::back_inserter(tokens),
                 [&tokens](std::string const &str)
  {
    std::string trimmedToken(str);
    boost::trim(trimmedToken);
    return std::move(trimmedToken);
  });

  return tokens;
}

inline
void TransposeTokens(std::vector<std::vector<std::string>> const &input,
                     std::vector<std::string> &output,
                     size_t index)
{
  output.clear();
  for(size_t i = 0; i != input.size(); ++i)
  {
    output.emplace_back(input[i][index]);
  }
}

} //namespace details

template<typename eT>
bool inline inplace_transpose(arma::Mat<eT>& X)
{
  try
  {
    X = arma::trans(X);
    return false;
  }
  catch (std::bad_alloc&)
  {
#if (ARMA_VERSION_MAJOR >= 4) || \
    ((ARMA_VERSION_MAJOR == 3) && (ARMA_VERSION_MINOR >= 930))
    arma::inplace_trans(X, "lowmem");
    return true;
#else
    Log::Fatal << "data::Load(): inplace_trans() is only available on Armadillo"
        << " 3.930 or higher. Ran out of memory to transpose matrix."
        << std::endl;
    return false;
#endif
  }
}

// Load column vector.
template<typename eT>
bool Load(const std::string& filename,
          arma::Col<eT>& vec,
          const bool fatal)
{
  return Load(filename, vec, fatal, false);
}

// Load row vector.
template<typename eT>
bool Load(const std::string& filename,
          arma::Row<eT>& rowvec,
          const bool fatal)
{
  return Load(filename, rowvec, fatal, false);
}

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          const bool transpose)
{
  Timer::Start("loading_data");

  // Get the extension.
  std::string extension = Extension(filename);

  // Catch nonexistent files by opening the stream ourselves.
  std::fstream stream;
#ifdef  _WIN32 // Always open in binary mode on Windows.
  stream.open(filename.c_str(), std::fstream::in | std::fstream::binary);
#else
  stream.open(filename.c_str(), std::fstream::in);
#endif
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

  if (extension == "csv" || extension == "tsv")
  {
    loadType = arma::diskio::guess_file_type(stream);
    if (loadType == arma::csv_ascii)
    {
      if (extension == "tsv")
        Log::Warn << "'" << filename << "' is comma-separated, not "
            "tab-separated!" << std::endl;
      stringType = "CSV data";
    }
    else if (loadType == arma::raw_ascii) // .csv file can be tsv.
    {
      if (extension == "csv")
      {
        // We should issue a warning, but we don't want to issue the warning if
        // there is only one column in the CSV (since there will be no commas
        // anyway, and it will be detected as arma::raw_ascii).
        const std::streampos pos = stream.tellg();
        std::string line;
        std::getline(stream, line, '\n');
        boost::trim(line);

        // Reset stream position.
        stream.seekg(pos);

        // If there are no spaces or whitespace in the line, then we shouldn't
        // print the warning.
        if ((line.find(' ') != std::string::npos) ||
            (line.find('\t') != std::string::npos))
        {
          Log::Warn << "'" << filename << "' is not a standard csv file."
              << std::endl;
        }
      }
      stringType = "raw ASCII formatted data";
    }
    else
    {
      unknownType = true;
      loadType = arma::raw_binary; // Won't be used; prevent a warning.
      stringType = "";
    }
  }
  else if (extension == "txt")
  {
    // This could be raw ASCII or Armadillo ASCII (ASCII with size header).
    // We'll let Armadillo do its guessing (although we have to check if it is
    // arma_ascii ourselves) and see what we come up with.

    // This is taken from load_auto_detect() in diskio_meat.hpp
    const std::string ARMA_MAT_TXT = "ARMA_MAT_TXT";
    //char* rawHeader = new char[ARMA_MAT_TXT.length() + 1];
    std::string rawHeader(ARMA_MAT_TXT.length(), '\0');
    std::streampos pos = stream.tellg();

    stream.read(&rawHeader[0], std::streamsize(ARMA_MAT_TXT.length()));
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (rawHeader == ARMA_MAT_TXT)
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
  }
  else if (extension == "bin")
  {
    // This could be raw binary or Armadillo binary (binary with header).  We
    // will check to see if it is Armadillo binary.
    const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";    
    std::string rawHeader(ARMA_MAT_BIN.length(), '\0');

    std::streampos pos = stream.tellg();

    stream.read(&rawHeader[0], std::streamsize(ARMA_MAT_BIN.length()));
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (rawHeader == ARMA_MAT_BIN)
    {
      stringType = "Armadillo binary formatted data";
      loadType = arma::arma_binary;
    }
    else // We can only assume it's raw binary.
    {
      stringType = "raw binary formatted data";
      loadType = arma::raw_binary;
    }    
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
  #if ARMA_VERSION_MAJOR == 4 && \
      (ARMA_VERSION_MINOR >= 300 && ARMA_VERSION_MINOR <= 400)
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Attempted to load '" << filename << "' as HDF5 data, but "
          << "Armadillo 4.300.0 through Armadillo 4.400.1 are known to have "
          << "bugs and one of these versions is in use.  Load failed."
          << std::endl;
    else
      Log::Warn << "Attempted to load '" << filename << "' as HDF5 data, but "
          << "Armadillo 4.300.0 through Armadillo 4.400.1 are known to have "
          << "bugs and one of these versions is in use.  Load failed."
          << std::endl;

    return false;
  #endif
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

  // We can't use the stream if the type is HDF5.
  bool success;
  if (loadType != arma::hdf5_binary)
    success = matrix.load(stream, loadType);
  else
    success = matrix.load(filename, loadType);

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

  // Now transpose the matrix, if necessary.  Armadillo loads HDF5 matrices
  // transposed, so we have to work around that.
  if (transpose && loadType != arma::hdf5_binary)
  {
    inplace_transpose(matrix);
  }
  else if (!transpose && loadType == arma::hdf5_binary)
  {
    inplace_transpose(matrix);
  }

  Timer::Stop("loading_data");

  // Finally, return the success indicator.
  return success;
}

// Load with mappings.  Unfortunately we have to implement this ourselves.
template<typename eT, typename PolicyType>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          DatasetMapper<PolicyType>& info,
          const bool fatal,
          const bool transpose)
{
  // Get the extension and load as necessary.
  Timer::Start("loading_data");

  // Get the extension.
  const std::string extension = Extension(filename);

  if (extension == "csv" || extension == "tsv" || extension == "txt")
  {
    LoadCSV loader(filename, fatal);
    loader.Load(matrix, info, transpose);
  }
  else if (extension == "arff")
  {
    Log::Info << "Loading '" << filename << "' as ARFF dataset.  "
        << std::flush;
    try
    {
      LoadARFF(filename, matrix, info);

      // We transpose by default.  So, un-transpose if necessary...
      if (!transpose)
        inplace_transpose(matrix);
    }
    catch (std::exception& e)
    {
      if (fatal)
        Log::Fatal << e.what() << std::endl;
      else
        Log::Warn << e.what() << std::endl;
    }
  }
  else
  {
    // The type is unknown.
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Unable to detect type of '" << filename << "'; "
          << "incorrect extension?" << std::endl;
    else
      Log::Warn << "Unable to detect type of '" << filename << "'; load failed."
          << " Incorrect extension?" << std::endl;

    return false;
  }

  Log::Info << "Size is " << (transpose ? matrix.n_cols : matrix.n_rows)
      << " x " << (transpose ? matrix.n_rows : matrix.n_cols) << ".\n";

  Timer::Stop("loading_data");

  return true;
}

} // namespace data
} // namespace mlpack

#endif
