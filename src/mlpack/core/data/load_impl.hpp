/**
 * @file core/data/load_impl.hpp
 * @author Ryan Curtin
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
#include "detect_file_type.hpp"
#include "string_algorithms.hpp"

namespace mlpack {
namespace data {

namespace details{

template<typename Tokenizer>
std::vector<std::string> ToTokens(Tokenizer& lineTok)
{
  std::vector<std::string> tokens;
  std::transform(std::begin(lineTok), std::end(lineTok),
                 std::back_inserter(tokens),
                 [&tokens](std::string const &str)
  {
    std::string trimmedToken(str);
    Trim(trimmedToken);
    return std::move(trimmedToken);
  });

  return tokens;
}

inline
void TransposeTokens(std::vector<std::vector<std::string>> const &input,
                     std::vector<std::string>& output,
                     size_t index)
{
  output.clear();
  for (size_t i = 0; i != input.size(); ++i)
  {
    output.emplace_back(input[i][index]);
  }
}

} // namespace details

template <typename MatType>
bool inline inplace_transpose(MatType& X, bool fatal)
{
  try
  {
    X = trans(X);
    return true;
  }
  catch (const std::exception& e)
  {
    if (fatal)
      Log::Fatal << "\nTranspose Operation Failed.\n"
          "Exception: " << e.what() << std::endl;
    else
      Log::Warn << "\nTranspose Operation Failed.\n"
          "Exception: " << e.what() << std::endl;

    return false;
  }
}

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  Timer::Start("loading_data");

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

  FileType loadType = inputLoadType;
  std::string stringType;
  if (inputLoadType == FileType::AutoDetect)
  {
    // Attempt to auto-detect the type from the given file.
    loadType = AutoDetect(stream, filename);
    // Provide error if we don't know the type.
    if (loadType == FileType::FileTypeUnknown)
    {
      Timer::Stop("loading_data");
      if (fatal)
        Log::Fatal << "Unable to detect type of '" << filename << "'; "
            << "incorrect extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << " failed. Incorrect extension?" << std::endl;

      return false;
    }
  }

  stringType = GetStringType(loadType);

#ifndef ARMA_USE_HDF5
  if (inputLoadType == FileType::HDF5Binary)
  {
    // Ensure that HDF5 is supported.
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
  }
#endif

  // Try to load the file; but if it's raw_binary, it could be a problem.
  if (loadType == FileType::RawBinary)
    Log::Warn << "Loading '" << filename << "' as " << stringType << "; "
        << "but this may not be the actual filetype!" << std::endl;
  else
    Log::Info << "Loading '" << filename << "' as " << stringType << ".  "
        << std::flush;

  // We can't use the stream if the type is HDF5.
  bool success;
  LoadCSV loader;
  
  if (loadType != FileType::HDF5Binary)
  {
    if (loadType == FileType::CSVASCII)
      success = loader.LoadNumericCSV(matrix, stream);
    else
      success = matrix.load(stream, ToArmaFileType(loadType));
  }
  else
    success = matrix.load(filename, ToArmaFileType(loadType));

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
    success = inplace_transpose(matrix, fatal);
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
    Log::Info << "Loading '" << filename << "' as CSV dataset.  " << std::flush;
    try
    {
      LoadCSV loader(filename);
      loader.LoadCategoricalCSV(matrix, info, transpose);
    }
    catch (std::exception& e)
    {
      Timer::Stop("loading_data");
      if (fatal)
        Log::Fatal << e.what() << std::endl;
      else
        Log::Warn << e.what() << std::endl;

      return false;
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
      {
        return inplace_transpose(matrix, fatal);
      }
    }
    catch (std::exception& e)
    {
      Timer::Stop("loading_data");
      if (fatal)
        Log::Fatal << e.what() << std::endl;
      else
        Log::Warn << e.what() << std::endl;

      return false;
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

// For loading data into sparse matrix
template <typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
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

  if (extension == "tsv" || extension == "txt")
  {
    loadType = arma::coord_ascii;
    stringType = "Coordinate Formatted Data for Sparse Matrix";
  }
  else if (extension == "bin")
  {
    // This could be raw binary or Armadillo binary (binary with header).  We
    // will check to see if it is Armadillo binary.
    const std::string ARMA_SPM_BIN = "ARMA_SPM_BIN";
    std::string rawHeader(ARMA_SPM_BIN.length(), '\0');

    std::streampos pos = stream.tellg();

    stream.read(&rawHeader[0], std::streamsize(ARMA_SPM_BIN.length()));
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (rawHeader == ARMA_SPM_BIN)
    {
      stringType = "Armadillo binary formatted data for sparse matrix";
      loadType = arma::arma_binary;
    }
    else // We can only assume it's raw binary.
    {
      stringType = "raw binary formatted data";
      loadType = arma::raw_binary;
    }
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

  bool success;

  success = matrix.load(stream, loadType);

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
    success = inplace_transpose(matrix, fatal);
  }

  Timer::Stop("loading_data");

  // Finally, return the success indicator.
  return success;
}

} // namespace data
} // namespace mlpack

#endif
