/**
 * @file core/data/load_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Data options, all possible options to load different data types and format
 * with specific settings into mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATA_OPTIONS_HPP
#define MLPACK_CORE_DATA_DATA_OPTIONS_HPP

#include <mlpack/prereqs.hpp>
#include <string>

#include "types.hpp"
#include "dataset_mapper.hpp"
#include "format.hpp"
#include "image_info.hpp"

namespace mlpack {
namespace data {

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
      bool noTranspose = false,
      FileType fileFormat = FileType::AutoDetect) :
    fatal(fatal),
    noTranspose(noTranspose),
    fileFormat(fileFormat)
  {
    // Do nothing.
  }

  // These are accessible to users.
  //! Get the error it it is fatal or not.
  const bool& Fatal() const { return fatal; }

  //! Modify the error to be fatal.
  bool& Fatal() { return fatal; }

  //! Get if the matrix is transposed or not.
  const bool& NoTranspose() const { return noTranspose; }

  //! Transpose the matrix if necessary.
  bool& NoTranspose() { return noTranspose; }

  //! Get the FileType.
  const FileType& FileFormat() const { return fileFormat; }

  //! Modify the FileType.
  FileType& FileFormat() { return fileFormat; }

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

  //! Public options.
  bool fatal;
  bool noTranspose;
  FileType fileFormat;
};

class ModelOptions : public DataOptions
{
 public:

   ModelOptions(
       bool model = false,
       format dataFormat = format::binary,
       std::string objectName = "") :
    model(model),
    dataFormat(dataFormat),
    objectName(objectName)
  {
    // Do Nothing.
  }

  explicit ModelOptions(const DataOptions& opts) :
    DataOptions(opts),
    model(false),
    dataFormat(format::binary),
    objectName("")
  {
    // Do Nothing.
  }

  //! Get if we are loading an model.
  const bool& Model() const { return model; }

  //! Modify if we are loading an model.
  bool& Model() { return model; }

  //! Get the FileType.
  const format& DataFormat() const { return dataFormat; }

  //! Modify the FileType.
  format& DataFormat() { return dataFormat; }

  //! Get if we are loading an image.
  const std::string& ObjectName() const { return objectName; }

  //! Modify if we are loading an image.
  std::string& ObjectName() { return objectName; }

 private:

  bool model;
  format dataFormat;
  std::string objectName;
};

class ImageOptions : public DataOptions
{
 public:
 
   ImageOptions(bool image = false) :
    image(image)
  {
    // Do Nothing.
  }
  //! Get if we are loading an image.
  const bool& Image() const { return image; }

  //! Modify if we are loading an image.
  bool& Image() { return image; }

  //! Get the FileType.
  const ImageInfo& ImageInfos() const { return imgInfo; }

  //! Modify the ImageInfo.
  ImageInfo& ImageInfos() { return imgInfo; }

 private:

  bool image;
  ImageInfo imgInfo;
};

class CSVOptions : public DataOptions
{

 public:
   CSVOptions(
       bool hasHeaders = false,
       bool semiColon = false,
       bool missingToNan = false,
       bool categorical = false,
       bool timeseries = false,
       int samplingRate = 0) :
     hasHeaders(hasHeaders),
     semiColon(semiColon),
     missingToNan(missingToNan),
     categorical(categorical),
     timeseries(timeseries),
     samplingRate(samplingRate)
  {
    // Do Nothing.
  }

  explicit CSVOptions(const DataOptions& opts) :
        DataOptions(opts),
        hasHeaders(false),
        semiColon(false),
        missingToNan(false),
        categorical(false),
        timeseries(false)
  {
    // Do Nothing.
  }

  //! Get if the dataset hasHeaders or not.
  const bool& HasHeaders() const { return hasHeaders; }

  //! Modify the dataset if it hasHeaders.
  bool& HasHeaders() { return hasHeaders; }

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

  //! Get if the labelCol exists.
  const bool& LabelCol() const { return labelCol; }

  //! Modify if we have labelCol data in the dataset.
  bool& LabelCol() { return labelCol; }

  //! Get if the timestampCol exists.
  const bool& TimestampCol() const { return timestampCol; }

  //! Modify if we have timestampCol data in the dataset.
  bool& TimestampCol() { return timestampCol; }

  //! Get if we are loading an timeseries.
  const bool& Timeseries() const { return timeseries; }

  //! Modify if we are loading an timeseries.
  bool& Timeseries() { return timeseries; }

  //! Get the Sampling rate.
  const int& SamplingRate() const { return samplingRate; }

  //! Set the sampling rate in HZ
  void SamplingRate(int hz) { samplingRate = hz; }

  //! Get the Sampling rate.
  const int& WindowSize() const { return windowSize; }

  //! Set the sampling rate in the time unit extracted from Sampling Rate.
  // If Sampling Rate is in kHz then Window size is in ms.
  // If Sampling Rate is in Hz then Window size is in s, etc.
  void WindowSize(int size) { windowSize = size; }

  //! Get the headers.
  const arma::field<std::string>& Headers() const { return headers; }

  //! Modify the headers.
  arma::field<std::string>& Headers() { return headers; }

  //! Get the FileType.
  const DatasetInfo& Mapper() const { return mapper; }

  //! Modify the DatasetMapper.
  DatasetInfo & Mapper() { return mapper; }

 private:
  bool hasHeaders;
  bool semiColon;
  bool missingToNan;
  bool categorical;
  bool timeseries;
  bool timestampCol;
  bool labelCol;
  int samplingRate;
  int windowSize;
  arma::field<std::string> headers;
  DatasetInfo mapper;
};

inline CSVOptions operator|(const CSVOptions& a, CSVOptions& b)
{
  CSVOptions output;
  output.SemiColon() = a.SemiColon() | b.SemiColon();
  output.MissingToNan() = a.MissingToNan() | b.MissingToNan();
  output.Categorical() = a.Categorical() | b.Categorical();
  output.Timeseries() = a.Timeseries() | b.Timeseries();
  return output;
}

inline ModelOptions operator|(const ModelOptions& a, const ModelOptions& b)
{
  ModelOptions output;
  output.Model() = a.Model() | b.Model();

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

inline ImageOptions operator|(const ImageOptions& a, const ImageOptions& b)
{
  ImageOptions output;

  output.Image() = a.Image() | b.Image();

  return output;
}

inline DataOptions operator|(const DataOptions& a, const DataOptions& b)
{
  DataOptions output;
  output.Fatal() = a.Fatal() | b.Fatal();
  output.NoTranspose() = a.NoTranspose() | b.NoTranspose();

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

  return output;
}

namespace DataOptionsTypes
{

struct FatalOptions: public DataOptions
{
  inline FatalOptions() : DataOptions() { this->Fatal() = true; }
};

struct NoFatalOptions: public DataOptions
{
  inline NoFatalOptions() : DataOptions() { this->Fatal() = false; }
};

struct TransposeOptions : public DataOptions
{
  inline TransposeOptions() : DataOptions() { this->NoTranspose() = false; }
};

struct NoTransposeOptions : public DataOptions
{
  inline NoTransposeOptions() : DataOptions() { this->NoTranspose() = true; }
};

struct HasHeadersOptions : public CSVOptions
{
  inline HasHeadersOptions() : CSVOptions() { this->HasHeaders() = true; }
};

struct SemiColonOptions : public CSVOptions
{
  inline SemiColonOptions() : CSVOptions() { this->SemiColon() = true; }
};

struct MissingToNanOptions : public CSVOptions
{
  inline MissingToNanOptions() : CSVOptions()
  {
    this->MissingToNan() = true;
  }
};

struct CategoricalOptions : public CSVOptions
{
  inline CategoricalOptions() : CSVOptions() { this->Categorical() = true; }
};

//! Data serialization options 
struct AutodetectOptions : public ModelOptions
{
  inline AutodetectOptions() : ModelOptions()
  {
    this->DataFormat() = format::autodetect;
  }
};

struct JsonModelOptions : public ModelOptions
{
  inline JsonModelOptions() : ModelOptions()
  {
    this->DataFormat() = format::json;
  }
};

struct XmlModelOptions : public ModelOptions
{
  inline XmlModelOptions() : ModelOptions()
  {
    this->DataFormat() = format::xml;
  }
};

struct BinaryModelOptions : public ModelOptions
{
  inline BinaryModelOptions() : ModelOptions()
  {
    this->DataFormat() = format::binary;
  }
};

//! File serialization options 
struct FileAutoDetectOptions : public DataOptions
{
  inline FileAutoDetectOptions() : DataOptions()
  {
    this->FileFormat() = FileType::AutoDetect;
  }
};

struct CSVOptions : public DataOptions
{
  inline CSVOptions() : DataOptions()
  {
    this->FileFormat() = FileType::CSVASCII;
  }
};

struct PGMOptions : public DataOptions
{
  inline PGMOptions() : DataOptions()
  {
    this->FileFormat() = FileType::PGMBinary;
  }
};

struct PPMOptions : public DataOptions
{
  inline PPMOptions() : DataOptions()
  {
    this->FileFormat() = FileType::PPMBinary;
  }
};

struct HDF5Options : public DataOptions
{
  inline HDF5Options() : DataOptions()
  {
    this->FileFormat() = FileType::HDF5Binary;
  }
};

struct ArmaASCIIOptions : public DataOptions
{
  inline ArmaASCIIOptions() : DataOptions()
  {
    this->FileFormat() = FileType::ArmaASCII;
  }
};

struct ArmaBinOptions : public DataOptions
{
  inline ArmaBinOptions() : DataOptions()
  {
    this->FileFormat() = FileType::ArmaBinary;
  }
};

struct RawASCIIOptions : public DataOptions
{
  inline RawASCIIOptions() : DataOptions()
  {
    this->FileFormat() = FileType::RawASCII;
  }
};

struct RawBinOptions : public DataOptions
{
  inline RawBinOptions() : DataOptions()
  {
    this->FileFormat() = FileType::RawBinary;
  }
};

struct CoordASCIIOptions : public DataOptions
{
  inline CoordASCIIOptions() : DataOptions()
  {
    this->FileFormat() = FileType::CoordASCII;
  }
};

} // namespace DataOptionsTypes

//! Boolean options
static const DataOptionsTypes::FatalOptions         Fatal;
static const DataOptionsTypes::NoFatalOptions       NoFatal;
static const DataOptionsTypes::HasHeadersOptions    HasHeaders;
static const DataOptionsTypes::TransposeOptions     Transpose;
static const DataOptionsTypes::NoTransposeOptions   NoTranspose;
static const DataOptionsTypes::SemiColonOptions     SemiColon;
static const DataOptionsTypes::MissingToNanOptions  MissingToNan;
static const DataOptionsTypes::CategoricalOptions   Categorical;

//! File options
static const DataOptionsTypes::CSVOptions            CSV;
static const DataOptionsTypes::PGMOptions            PGM_BIN;
static const DataOptionsTypes::PPMOptions            PPM_BIN;
static const DataOptionsTypes::HDF5Options           HDF5_BIN;
static const DataOptionsTypes::ArmaASCIIOptions      ARMA_ASCII;
static const DataOptionsTypes::ArmaBinOptions        ARMA_BIN;
static const DataOptionsTypes::RawASCIIOptions       RAW_ASCII;
static const DataOptionsTypes::RawBinOptions         BIN_ASCII;
static const DataOptionsTypes::CoordASCIIOptions     COORD_ASCII;
static const DataOptionsTypes::FileAutoDetectOptions AutoDetect_File;

//! Data serialization options 
static const DataOptionsTypes::AutodetectOptions    AutoDetect_SER;
static const DataOptionsTypes::JsonModelOptions     JSON_SER;
static const DataOptionsTypes::XmlModelOptions      XML_SER;
static const DataOptionsTypes::BinaryModelOptions   BIN_SER;


} // namespace data
} // namespace mlpack

#endif
