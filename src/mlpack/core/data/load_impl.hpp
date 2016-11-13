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
#include "extension.hpp"

#include <algorithm>
#include <mlpack/core/util/timers.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include "serialization_shim.hpp"

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
void TransPoseTokens(std::vector<std::vector<std::string>> const &input,
                     std::vector<std::string> &output,
                     size_t index)
{
  output.clear();
  for(size_t i = 0; i != input.size(); ++i)
  {
    output.emplace_back(input[i][index]);
  }
}

}

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
  std::string extension = Extension(filename);

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

  if (extension == "csv" || extension == "tsv" || extension == "txt")
  {
    // True if we're looking for commas; if false, we're looking for spaces.
    bool commas = (extension == "csv");

    std::string type;
    if (extension == "csv")
      type = "CSV data";
    else
      type = "raw ASCII-formatted data";

    Log::Info << "Loading '" << filename << "' as " << type << ".  "
        << std::flush;
    std::string separators;
    if (commas)
      separators = ",";
    else
      separators = " \t";

    // We'll load this as CSV (or CSV with spaces or tabs) according to
    // RFC4180.  So the first thing to do is determine the size of the matrix.
    std::string buffer;
    size_t cols = 0;

    std::getline(stream, buffer, '\n');
    // Count commas and whitespace in the line, ignoring anything inside
    // quotes.
    typedef boost::tokenizer<boost::escaped_list_separator<char>> Tokenizer;
    boost::escaped_list_separator<char> sep("\\", separators, "\"");
    Tokenizer tok(buffer, sep);
    for (Tokenizer::iterator i = tok.begin(); i != tok.end(); ++i)
      ++cols;

    // Now count the number of lines in the file.  We've already counted the
    // first one.
    size_t rows = 1;
    while (!stream.eof() && !stream.bad() && !stream.fail())
    {
      std::getline(stream, buffer, '\n');
      if (!stream.fail())
        ++rows;
    }

    // Now we have the size.  So resize our matrix.
    if (transpose)
    {
      matrix.set_size(cols, rows);
      info = DatasetMapper<PolicyType>(info.Policy(), cols);
    }
    else
    {
      matrix.set_size(rows, cols);
      info = DatasetMapper<PolicyType>(info.Policy(), rows);
    }

    stream.close();
    stream.open(filename, std::fstream::in);

    if (transpose)
    {
      std::vector<std::vector<std::string>> tokensArray;
      std::vector<std::string> tokens;
      while (!stream.bad() && !stream.fail() && !stream.eof())
      {
        // Extract line by line.
        std::getline(stream, buffer, '\n');
        Tokenizer lineTok(buffer, sep);
        tokens = details::ToTokens(lineTok);
        if (tokens.size() == cols)
        {
          tokensArray.emplace_back(std::move(tokens));
        }
      }
      for(size_t i = 0; i != cols; ++i)
      {
        details::TransPoseTokens(tokensArray, tokens, i);
        info.MapTokens(tokens, i, matrix);
      }
    }
    else
    {
      size_t row = 0;
      while (!stream.bad() && !stream.fail() && !stream.eof())
      {
        // Extract line by line.
        std::getline(stream, buffer, '\n');
        Tokenizer lineTok(buffer, sep);
        info.MapTokens(details::ToTokens(lineTok), row, matrix);
        ++row;
      }
    }
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
  std::ifstream ifs;
#ifdef _WIN32 // Open non-text in binary mode on Windows.
  if (f == format::binary)
    ifs.open(filename, std::ifstream::in | std::ifstream::binary);
  else
    ifs.open(filename, std::ifstream::in);
#else
  ifs.open(filename, std::ifstream::in);
#endif

  if (!ifs.is_open())
  {
    if (fatal)
      Log::Fatal << "Unable to open file '" << filename << "' to load object '"
          << name << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "' to load object '"
          << name << "'." << std::endl;

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
