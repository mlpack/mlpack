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
#include "format.hpp"

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
 * All possible DataOptions grouped under one class.
 * This will allow us to have consistent API over the next mlpack versions. If
 * new data options might be necessary, then they should be added in the
 * following.
 */
class LoadOptions
{
 public:
  /**
   * DataOptions constructor that takes the followint parameters.
   *  - fatal : true if fatal error should be reported.
   *  - hasHeaders : true if the dataset file has headers we need to recover
   *  - transpose: true by default, Transpose armadillo matrix to column major
   *  - semiColon: If the dataset separator is a semicolon instead of commas.
   *  - missingToNan: replace missing values to NaN.
   *  - categorical: if the dataset contains categorical values.
   *  - image: true if we are trying to load an image.
   *  - fileFormat: Detect automaticaly the file type, csv, txt, hdf5.
   *  - objectName: Model / object name that is going to be serialized
   *  - imgInfo: Information about the image if we already have them.
   *  - dataFormat: the data serialization format: xml, bin, json
   */
  LoadOptions(
      bool fatal = true,
      bool hasHeaders = false,
      bool transpose = true,
      bool semiColon = false,
      bool missingToNan = false,
      bool categorical = false,
      bool image = false,
      bool model = false,
      FileType fileFormat = FileType::AutoDetect,
      format dataFormat = format::binary,
      std::string objectName = "") :
    fatal(fatal),
    hasHeaders(hasHeaders),
    transpose(transpose),
    semiColon(semiColon),
    missingToNan(missingToNan),
    categorical(categorical),
    image(image),
    model(model),
    fileFormat(fileFormat),
    dataFormat(dataFormat),
    objectName(objectName)
  {
    // Do nothing.
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

  //! Get if the categorical data exists.
  const bool& Categorical() const { return categorical; }

  //! Modify if we have categorical data in the dataset.
  bool& Categorical() { return categorical; }

  //! Get if we are loading an image.
  const bool& Image() const { return image; }

  //! Modify if we are loading an image.
  bool& Image() { return image; }

  //! Get if we are loading an model.
  const bool& Model() const { return model; }

  //! Modify if we are loading an model.
  bool& Model() { return model; }

  //! Get the headers.
  const arma::field<std::string>& Headers() const { return headers; }

  //! Modify the headers.
  arma::field<std::string>& Headers() { return headers; }

  //! Get the FileType.
  const FileType& FileFormat() const { return fileFormat; }

  //! Modify the FileType.
  FileType& FileFormat() { return fileFormat; }

  //! Get the FileType.
  const format& DataFormat() const { return dataFormat; }

  //! Modify the FileType.
  format& DataFormat() { return dataFormat; }

  //! Get the FileType.
  const ImageInfo& ImageInfos() const { return imgInfo; }

  //! Modify the ImageInfo.
  ImageInfo& ImageInfos() { return imgInfo; }

  //! Get the FileType.
  const DatasetInfo& Mapper() const { return mapper; }

  //! Modify the DatasetMapper.
  DatasetInfo & Mapper() { return mapper; }

  //! Get if we are loading an image.
  const std::string& ObjectName() const { return objectName; }

  //! Modify if we are loading an image.
  std::string& ObjectName() { return objectName; }

  /**
   * Given a file type, return a logical name corresponding to that file type.
   */
  const std::string FileTypeToString() const
  {
    switch (fileFormat)
    {
      case FileType::CSVASCII:    return "CSV data";
      case FileType::RawASCII:    return "raw ASCII formatted data";
      case FileType::RawBinary:   return "raw binary formatted data";
      case FileType::ArmaASCII:   return "Armadillo ASCII formatted data";
      case FileType::ArmaBinary:  return "Armadillo binary formatted data";
      case FileType::PGMBinary:   return "PGM data";
      case FileType::HDF5Binary:  return "HDF5 data";
      case FileType::CoordASCII:  return "ASCII formatted sparse coordinate data";
      default:                    return "";
    }
  }

 private:

  bool fatal;
  bool hasHeaders;
  bool transpose;
  bool semiColon;
  bool missingToNan;
  bool categorical;
  bool image;
  bool model;
  FileType fileFormat;
  format dataFormat;
  std::string objectName;
  arma::field<std::string> headers;
  ImageInfo imgInfo;
  DatasetInfo mapper;
};

class FatalOptions: public LoadOptions
{
 public:
  inline FatalOptions() : LoadOptions() { this->Fatal() = true; }
};

class HasHeadersOptions : public LoadOptions
{
 public:
  inline HasHeadersOptions() : LoadOptions() { this->HasHeaders() = true; }
};

class TransposeOptions : public LoadOptions
{
 public:
  inline TransposeOptions() : LoadOptions() { this->Transpose() = true; }
};

class SemiColonOptions : public LoadOptions
{
 public:
  inline SemiColonOptions() : LoadOptions() { this->SemiColon() = true; }
};

class MissingToNanOptions : public LoadOptions
{
 public:
  inline MissingToNanOptions() : LoadOptions() { this->MissingToNan() = true; }
};

class CategoricalOptions : public LoadOptions
{
 public:
  inline CategoricalOptions() : LoadOptions() { this->Categorical() = true; }
};

class ImageOptions : public LoadOptions
{
 public:
  inline ImageOptions() : LoadOptions() { this->Image() = true; }
};

class ModelOptions : public LoadOptions
{
 public:
  inline ModelOptions() : LoadOptions() { this->Model() = true; }
};

//! Data serialization options 
class AutodetectOptions : public LoadOptions
{
 public:
  inline AutodetectOptions() : LoadOptions() { this->DataFormat() = format::autodetect; }
};

class JsonDataOptions : public LoadOptions
{
 public:
  inline JsonDataOptions() : LoadOptions() { this->DataFormat() = format::json; }
};

class XmlDataOptions : public LoadOptions
{
 public:
  inline XmlDataOptions() : LoadOptions() { this->DataFormat() = format::xml; }
};

class BinaryDataOptions : public LoadOptions
{
 public:
  inline BinaryDataOptions() : LoadOptions() { this->DataFormat() = format::binary; }
};

//! File serialization options 
class FileAutoDetectOptions : public LoadOptions
{
 public:
  inline FileAutoDetectOptions() : LoadOptions() { this->FileFormat() = FileType::AutoDetect; }
};

class CSVOptions : public LoadOptions
{
 public:
   inline CSVOptions() : LoadOptions() { this->FileFormat() = FileType::CSVASCII; }
};

class PGMOptions : public LoadOptions
{
 public:
   inline PGMOptions() : LoadOptions() { this->FileFormat() = FileType::PGMBinary; }
};

class PPMOptions : public LoadOptions
{
 public:
   inline PPMOptions() : LoadOptions() { this->FileFormat() = FileType::PPMBinary; }
};

class HDF5Options : public LoadOptions
{
 public:
   inline HDF5Options() : LoadOptions() { this->FileFormat() = FileType::HDF5Binary; }
};

class ArmaASCIIOptions : public LoadOptions
{
 public:
   inline ArmaASCIIOptions() : LoadOptions() { this->FileFormat() = FileType::ArmaASCII; }
};

class ArmaBinOptions : public LoadOptions
{
 public:
   inline ArmaBinOptions() : LoadOptions() { this->FileFormat() = FileType::ArmaBinary; }
};

class RawASCIIOptions : public LoadOptions
{
 public:
   inline RawASCIIOptions() : LoadOptions() { this->FileFormat() = FileType::RawASCII; }
};

class RawBinOptions : public LoadOptions
{
 public:
   inline RawBinOptions() : LoadOptions() { this->FileFormat() = FileType::RawBinary; }
};

class CoordASCIIOptions : public LoadOptions
{
 public:
   inline CoordASCIIOptions() : LoadOptions() { this->FileFormat() = FileType::CoordASCII; }
};

//! Boolean options
static const FatalOptions         Fatal;
static const HasHeadersOptions    HasHeaders;
static const TransposeOptions     Transpose;
static const SemiColonOptions     SemiColon;
static const MissingToNanOptions  MissingToNan;
static const CategoricalOptions   Categorical;
static const ImageOptions         Image;
static const ModelOptions         Model;

//! File options
static const CSVOptions            CSV; 
static const PGMOptions            PGM_BIN;
static const PPMOptions            PPM_BIN;
static const HDF5Options           HDF5_BIN;
static const ArmaASCIIOptions      ARMA_ASCII;
static const ArmaBinOptions        ARMA_BIN;
static const RawASCIIOptions       RAW_ASCII;
static const RawBinOptions         BIN_ASCII;
static const CoordASCIIOptions     COORD_ASCII;
static const FileAutoDetectOptions AutoDetect_File;

//! Data serialization options 
static const AutodetectOptions    AutoDetect_SER;
static const JsonDataOptions      JSON_SER;
static const XmlDataOptions       XML_SER;
static const BinaryDataOptions    BIN_SER;

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
bool DetectFileType(const std::string& filename,
                    std::fstream& stream,
                    FileType& loadType,
                    const LoadOptions& opts)
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
