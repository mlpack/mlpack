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
 * This will allow us to have consistent data API for mlpack. If new data
 * options might be necessary, then they should be added in the following.
 */
class DataOptions
{
 public:
  /**
   * DataOptions constructor that takes the followint parameters.
   *  - fatal : true if fatal error should be reported.
   *  - hasHeaders : false if the dataset file has headers we need to recover
   *  - noTranspose: false by default, Transpose armadillo matrix to column major
   *  - semiColon: If the dataset separator is a semicolon instead of commas.
   *  - missingToNan: replace missing values to NaN.
   *  - categorical: if the dataset contains categorical values.
   *  - image: true if we are trying to load an image.
   *  - fileFormat: Detect automaticaly the file type, csv, txt, hdf5.
   *  - objectName: Model / object name that is going to be serialized
   *  - imgInfo: Information about the image if we already have them.
   *  - dataFormat: the data serialization format: xml, bin, json
   */
  DataOptions(
      bool fatal = false,
      bool hasHeaders = false,
      bool noTranspose = false,
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
    noTranspose(noTranspose),
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
  const bool& NoTranspose() const { return noTranspose; }

  //! Transpose the matrix if necessary.
  bool& NoTranspose() { return noTranspose; }

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
  bool noTranspose;
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

inline DataOptions operator|(const DataOptions& a, const DataOptions& b)
{
  DataOptions output;
  output.Fatal() = a.Fatal() | b.Fatal();
  output.NoTranspose() = a.NoTranspose() | b.NoTranspose();
  output.SemiColon() = a.SemiColon() | b.SemiColon();
  output.MissingToNan() = a.MissingToNan() | b.MissingToNan();
  output.Categorical() = a.Categorical() | b.Categorical();
  output.Image() = a.Image() | b.Image();
  output.Model() = a.Model() | b.Model();

  if (a.FileFormat() == FileType::FileTypeUnknown)
  {
    output.FileFormat() = b.FileFormat();
    return output;
  }
  else if (b.FileFormat() == FileType::FileTypeUnknown)
  {
    output.FileFormat() = a.FileFormat();
    return output;
  }

  if (a.FileFormat() != b.FileFormat())
    throw std::runtime_error("File formats don't match!");
  else
    output.FileFormat() = a.FileFormat();

  if (a.DataFormat() == format::unknown)
  {
    output.DataFormat() = b.DataFormat();
    return output;
  }
  else if (b.DataFormat() == format::unknown)
  {
    output.DataFormat() = a.DataFormat();
    return output;
  }

  if (a.DataFormat() != b.DataFormat())
    throw std::runtime_error("Serialization formats don't match!");
  else
    output.DataFormat() = a.DataFormat();

  return output;
}

struct FatalOptions: public DataOptions
{
  inline FatalOptions() : DataOptions() { this->Fatal() = true; }
};

struct HasHeadersOptions : public DataOptions
{
  inline HasHeadersOptions() : DataOptions() { this->HasHeaders() = true; }
};

struct NoTransposeOptions : public DataOptions
{
  inline NoTransposeOptions() : DataOptions() { this->NoTranspose() = true; }
};

struct SemiColonOptions : public DataOptions
{
  inline SemiColonOptions() : DataOptions() { this->SemiColon() = true; }
};

struct MissingToNanOptions : public DataOptions
{
  inline MissingToNanOptions() : DataOptions() { this->MissingToNan() = true; }
};

struct CategoricalOptions : public DataOptions
{
  inline CategoricalOptions() : DataOptions() { this->Categorical() = true; }
};

struct ImageOptions : public DataOptions
{
  inline ImageOptions() : DataOptions() { this->Image() = true; }
};

struct ModelOptions : public DataOptions
{
  inline ModelOptions() : DataOptions() { this->Model() = true; }
};

//! Data serialization options 
struct AutodetectOptions : public DataOptions
{
  inline AutodetectOptions() : DataOptions() { this->DataFormat() = format::autodetect; }
};

struct JsonDataOptions : public DataOptions
{
  inline JsonDataOptions() : DataOptions() { this->DataFormat() = format::json; }
};

struct XmlDataOptions : public DataOptions
{
  inline XmlDataOptions() : DataOptions() { this->DataFormat() = format::xml; }
};

struct BinaryDataOptions : public DataOptions
{
  inline BinaryDataOptions() : DataOptions() { this->DataFormat() = format::binary; }
};

//! File serialization options 
struct FileAutoDetectOptions : public DataOptions
{
  inline FileAutoDetectOptions() : DataOptions() { this->FileFormat() = FileType::AutoDetect; }
};

struct CSVOptions : public DataOptions
{
  inline CSVOptions() : DataOptions() { this->FileFormat() = FileType::CSVASCII; }
};

struct PGMOptions : public DataOptions
{
  inline PGMOptions() : DataOptions() { this->FileFormat() = FileType::PGMBinary; }
};

struct PPMOptions : public DataOptions
{
  inline PPMOptions() : DataOptions() { this->FileFormat() = FileType::PPMBinary; }
};

struct HDF5Options : public DataOptions
{
  inline HDF5Options() : DataOptions() { this->FileFormat() = FileType::HDF5Binary; }
};

struct ArmaASCIIOptions : public DataOptions
{
  inline ArmaASCIIOptions() : DataOptions() { this->FileFormat() = FileType::ArmaASCII; }
};

struct ArmaBinOptions : public DataOptions
{
  inline ArmaBinOptions() : DataOptions() { this->FileFormat() = FileType::ArmaBinary; }
};

struct RawASCIIOptions : public DataOptions
{
  inline RawASCIIOptions() : DataOptions() { this->FileFormat() = FileType::RawASCII; }
};

struct RawBinOptions : public DataOptions
{
  inline RawBinOptions() : DataOptions() { this->FileFormat() = FileType::RawBinary; }
};

struct CoordASCIIOptions : public DataOptions
{
  inline CoordASCIIOptions() : DataOptions() { this->FileFormat() = FileType::CoordASCII; }
};

//! Boolean options
static const FatalOptions         Fatal;
static const HasHeadersOptions    HasHeaders;
static const NoTransposeOptions   NoTranspose;
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
