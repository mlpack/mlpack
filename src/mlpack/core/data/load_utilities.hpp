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

#include "dataset_mapper.hpp"
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

/**
 * All possible load options grouped under one struct.
 * This will allow us to have consistent API over the next mlpack versions. If
 * new load options might be necessary, then they should be added in the
 * following.
 * The supported options are the following:
 *  - fatal : true if fatal error should be reported.
 *  - hasHeaders : true if the dataset file has headers we need to recover
 *  - transpose: true by default, Transpose armadillo matrix to column major
 *  - arma::csv_opts: possible csv options provided by armadillo
 *  - arma::field<std::string> headers: contains the header of the dataset file
 *  - FileType format: Detect automaticaly the file type, csv, txt, hdf5.
 */
class LoadOptions
{
 public:
  /**
    *
    */
  LoadOptions() :
    fatal(fatal),
    hasHeaders(hasHeaders),
    transpose(transpose),
    semiColon(semiColon),
    missingToNan(missingToNan),
    headers(headers),
    format(format),
    categorical(categorical)
  {
    // We can not have these two policies to be set to true at the same time.
    // Therefore, a simple sanity check is added in here and reverse assignment.
    if (fillForward)
    {
      fillForward = true;
      fillBackward = false;
    }
    else if (fillBackward)
    {
      fillForward = false;
      fillBackward = true;
    }
  }

  //! Get the error it it is fatal or not.
  const bool& Fatal() const { return fatal; }

  //! Modify the error to be fatal.
  bool& Fatal() { return fatal; }

  //! Get if the dataset hasHeaders or not.
  const bool& HasHeaders() const { return hasHeaders; }

  //! Modify the dataset if it hasHeaders.
  bool& HasHeaders() { return hasHeaders; }

  //! Get if the matrix is transposed or not.
  const bool& Transpose() const { return transpose; }

  //! Transpose the matrix if necessary.
  bool& Transpose() { return transpose; }

  //! Get if the separator is a semicolon in the matrix.
  const bool& SemiColon() const { return semiColon; }

  //! Modify the separator type in the matrix.
  bool& SemiColon() { return semiColon; }

  //! Get if the separator is a semicolon in the matrix.
  const bool& MissingToNan() const { return missingToNan; }

  //! Modify the separator type in the matrix.
  bool& MissingToNan() { return missingToNan; }

  //! Get the FillForward policy if we have nan..
  const bool& FillForward() { return fillForward; }

  //! Modify the FiLlForward policy if we have nan
  bool& FillForward() { return fillForward; }

  //! Get the FillBackward policy if we have nan..
  const bool& FillBackward() { return fillBackward; }

  //! Modify the FiLlBackward policy if we have nan
  bool& FillBackward() { return fillBackward; }

  //! Get the headers.
  const arma::field<std::string>& Headers() const { return headers; }

  //! Modify the headers.
  arma::field<std::string>& Headers() { return headers; }

  //! Get the FileType.
  const FileType& FileFormat() const { return format; }

  //! Modify the FileType.
  FileType& FileFormat() { return format; }

  //! Get if the categorical data exists.
  const bool& Categorical() const { return categorical; }

  //! Modify if we have categorical data in the dataset.
  bool& Categorical() { return categorical; }

   //! Get the FileType.
  const ImageInfo& ImageInfos() const { return imgInfo; }

  //! Modify the ImageInfo.
  ImageInfo& ImageInfos() { return imgInfo; }

  //! Get if we are loading an image.
  const Image& Image() const { return image; }

  //! Modify if we are loading an image.
  Image& Image() { return image; }

  //! Get the FileType.
  const DatasetMapper& Mapper() const { return mapper; }

  //! Modify the DatasetMapper.
  DatasetMapper& Mapper() { return mapper; }

 private:

  bool fatal;
  bool hasHeaders;
  bool transpose;
  bool semiColon;
  bool missingToNan;
  bool fillForward;
  bool fillBackward;
  bool categorical;
  bool image;
  arma::field<std::string> headers;
  FileType format;
  ImageInfo imgInfo;
  DatasetMapper mapper;
};

inline
bool FileExist(const std::string& filename,
               std::fstream& stream,
               const LoadOptions& opts)
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
bool DetectFileType(const std::fstream& stream,
                    FileType& loadType,
                    const LoadOptions& opts);
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
