/**
 * @file core/data/load_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 * @author Gopi Tatiraju
 *
 * Implementation of templatized load() function defined in load.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_UTILITIES_HPP
#define MLPACK_CORE_DATA_LOAD_UTILITIES_HPP

#include <mlpack/prereqs.hpp>
#include <string>

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
bool FileExist(const std::string& filename,
               std::fstream& stream,
               const DataOptions& opts)
{
#ifdef  _WIN32 // Always open in binary mode on Windows.
  stream.open(filename.c_str(), std::fstream::in | std::fstream::binary);
#else
  stream.open(filename.c_str(), std::fstream::in);
#endif
  if (!stream.is_open())
  {
    Timer::Stop("loading_data");
    if (opts.Fatal())
      Log::Fatal << "Cannot open file '" << filename << "'. " << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "'; load failed."
          << std::endl;

    return false;
  }
  return true;
}

inline
bool DetectFileType(const std::string& filename,
                    std::fstream& stream,
                    FileType& loadType,
                    const DataOptions& opts)
{
  if (opts.FileFormat() == FileType::AutoDetect)
  {
    // Attempt to auto-detect the type from the given file.
    loadType = AutoDetect(stream, filename);
    // Provide error if we don't know the type.
    if (loadType == FileType::FileTypeUnknown)
    {
      Timer::Stop("loading_data");
      if (opts.Fatal())
        Log::Fatal << "Unable to detect type of '" << filename << "'; "
            << "incorrect extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << "failed. Incorrect extension?" << std::endl;

      return false;
    }
  }
  return true;
}

} //namespace data
} //namespace mlpack

#endif
