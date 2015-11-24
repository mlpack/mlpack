/**
 * @file load_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templatized load() function defined in load.hpp.
 */
#ifndef __MLPACK_CORE_DATA_LOAD_IMPL_HPP
#define __MLPACK_CORE_DATA_LOAD_IMPL_HPP

// In case it hasn't already been included.
#include "load.hpp"
#include "extension.hpp"

#include <algorithm>
#include <mlpack/core/util/timers.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "serialization_shim.hpp"

namespace mlpack {
namespace data {

template<typename eT>
bool inline inplace_transpose(arma::Mat<eT>& X)
{
  try
  {
    X = arma::trans(X);
    return false;
  }
  catch (std::bad_alloc& exception)
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

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          bool transpose)
{
  Timer::Start("loading_data");

  // Get the extension.
  std::string extension = Extension(filename);

  // Catch nonexistent files by opening the stream ourselves.
  std::fstream stream;
#ifdef  _WIN32
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
  {
    inplace_transpose(matrix);
  }

  Timer::Stop("loading_data");

  // Finally, return the success indicator.
  return success;
}

// Load a model from file.
template<typename T>
bool Load(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f)
{
  if (f == format::autodetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      f = format::xml;
    else if (extension == "bin")
      f = format::binary;
    else if (extension == "txt")
      f = format::text;
    else
    {
      if (fatal)
        Log::Fatal << "Unable to detect type of '" << filename << "'; incorrect"
            << " extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << "failed.  Incorrect extension?" << std::endl;

      return false;
    }
  }

  // Now load the given format.
  std::ifstream ifs(filename);
  if (!ifs.is_open())
  {
    if (fatal)
      Log::Fatal << "Unable to open file '" << filename << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "'." << std::endl;

    return false;
  }

  try
  {
    if (f == format::xml)
    {
      boost::archive::xml_iarchive ar(ifs);
      ar >> CreateNVP(t, name);
    }
    else if (f == format::text)
    {
      boost::archive::text_iarchive ar(ifs);
      ar >> CreateNVP(t, name);
    }
    else if (f == format::binary)
    {
      boost::archive::binary_iarchive ar(ifs);
      ar >> CreateNVP(t, name);
    }

    return true;
  }
  catch (boost::archive::archive_exception& e)
  {
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

} // namespace data
} // namespace mlpack

#endif
