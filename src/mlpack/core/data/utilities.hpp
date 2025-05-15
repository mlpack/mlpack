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

#include "detect_file_type.hpp"

namespace mlpack {
namespace data {

namespace details {

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

template<typename DataOptionsType>
bool OpenFile(const std::string& filename,
              DataOptionsType& opts,
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
  // Add here and else if for ModelOptions in a couple of stages.
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
    if (opts.Fatal() && isLoading)
      Log::Fatal << "Cannot open file '" << filename << "' for loading.  "
          << "Please check if the file exists." << std::endl;

    else if (!opts.Fatal() && isLoading)
      Log::Warn << "Cannot open file '" << filename << "' for loading.  "
          << "Please check if the file exists." << std::endl;

    else if (opts.Fatal() && !isLoading)
      Log::Fatal << "Cannot open file '" << filename << "' for saving.  "
          << "Please check if you have permissions for writing." << std::endl;

    else if (!opts.Fatal() && !isLoading)
      Log::Warn << "Cannot open file '" << filename << "' for saving.  "
          << "Please check if you have permissions for writing." << std::endl;

    return false;
  }
  return true;
}

template<typename MatType, typename DataOptionsType>
bool DetectFileType(const std::string& filename,
                    DataOptionsType& opts,
                    bool isLoading,
                    std::fstream* stream = nullptr)
{
  // Add if for ModelOptions in a couple of stages
  if (opts.Format() == FileType::AutoDetect)
  {
    if (isLoading)
      // Attempt to auto-detect the type from the given file.
      opts.Format() = AutoDetect(*stream, filename);
    else
      DetectFromExtension<MatType>(filename, opts);
    // Provide error if we don't know the type.
    if (opts.Format() == FileType::FileTypeUnknown)
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
  return true;
}

template<typename MatType, typename DataOptionsType>
bool SaveMatrix(const MatType& matrix,
                const DataOptionsType& opts,
                const std::string& filename,
                std::fstream& stream)
{
  bool success = false;
  if (opts.Format() == FileType::HDF5Binary)
  {
#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    success = matrix.save(filename, ToArmaFileType(opts.Format()))
#endif
  }
  else
  {
    success = matrix.save(stream, ToArmaFileType(opts.Format()));
  }
  return success;
}

} //namespace data
} //namespace mlpack

#endif
