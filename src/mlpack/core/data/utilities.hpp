/**
 * @file core/data/utilities.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 * @author Gopi Tatiraju
 *
 * Utilities functions that can be used during loading and saving the data..
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_UTILITIES_HPP
#define MLPACK_CORE_DATA_UTILITIES_HPP

#include <mlpack/prereqs.hpp>
#include <string>

#include "detect_file_type.hpp"

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

inline
bool OpenFile(const std::string& filename,
              DataOptions& opts,
              bool isLoading,
              std::fstream& stream)
{
  if (isLoading)
  {
#ifdef  _WIN32 // Always open in binary mode on Windows.
    stream.open(filename.c_str(), std::fstream::in
        | std::fstream::binary);
#else
    stream.open(filename.c_str(), std::fstream::in);
#endif
  }
  else if (opts.Model())
  {
#ifdef _WIN32 // Open non-text types in binary mode on Windows.
  if (opts.DataFormat() == format::binary)
    stream.open(filename, std::fstream::out
        | std::streamtream::binary);
  else
    stream.open(filename, std::fstream::out);
#else
  stream.open(filename, std::fstream::out);
#endif
  }
  else 
  {
#ifdef  _WIN32 // Always open in binary mode on Windows.
    stream.open(filename.c_str(), std::fstream::out
        | std::fstream::binary);
#else
    stream.open(filename.c_str(), std::fstream::out);
#endif
  } 

  if (!stream.is_open())
  {
    if (opts.Fatal())
      Log::Fatal << "Cannot open file '" << filename << "'. \n"
          << "please check if the file is available if loading. \n" 
          << "or if you have the rights for writing if saving. \n";

    else
      Log::Warn << "Cannot open file '" << filename << "'. \n"
          << "please check if the file is available if loading. \n" 
          << "or if you have the rights for writing if saving. \n";

    return false;
  }
  return true;
}

inline
bool DetectFileType(const std::string& filename,
                    DataOptions& opts,
                    bool isLoading,
                    std::fstream* stream = nullptr)
{
  if (opts.Model())
  {
    if (opts.DataFormat() == format::autodetect)
    {
      DetectFromExtension(filename, opts);
      if (opts.DataFormat() != format::xml || opts.DataFormat() != format::json
          || opts.DataFormat() != format::binary)
      {
        if (opts.Fatal())
          Log::Fatal << "Unable to detect type of '" << filename << "'; incorrect"
              << " extension? (allowed: xml/bin/json)" << std::endl;
        else
          Log::Warn << "Unable to detect type of '" << filename << "'; save "
              << "failed.  Incorrect extension? (allowed: xml/bin/json)"
              << std::endl;

        return false;
      }
    }
  }
  else
  {
    if (opts.FileFormat() == FileType::AutoDetect)
    {
      if (isLoading)
        // Attempt to auto-detect the type from the given file.
        opts.FileFormat() = AutoDetect(*stream, filename);
      else 
        DetectFromExtension(filename, opts);
      // Provide error if we don't know the type.
      if (opts.FileFormat() == FileType::FileTypeUnknown)
      {
        if (opts.Fatal())
          Log::Fatal << "Unable to detect type of '" << filename << "'; "
              << "Incorrect extension?" << std::endl;
        else
          Log::Warn << "Unable to detect type of '" << filename << "'; "
              << "Incorrect extension?" << std::endl;

        return false;
      }
    }
  }
  return true;
}

template<typename MatType>
bool SaveMatrix(const MatType& matrix, DataOptions& opts, std::fstream& stream)
{
  bool success = false;
  if (opts.FileFormat() == FileType::HDF5Binary)
  {
#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    success = matrix.save(filename, ToArmaFileType(opts.FileFormat()))
#endif
  }
  else
  {
    success = matrix.save(stream, ToArmaFileType(opts.FileFormat()));
  }
  return success;
}

} //namespace data
} //namespace mlpack

#endif
