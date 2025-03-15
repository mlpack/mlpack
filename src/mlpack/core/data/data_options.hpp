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

#include "dataset_mapper.hpp"
#include "detect_file_type.hpp"
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
    timeseries(timeseries),
    fileFormat(fileFormat),
    dataFormat(dataFormat),
    objectName(objectName)
  {
    // Do nothing.
  }

  //! Get if it is load or not.
  const bool& Load() const { return load; }

  //! Modify to be load.
  bool& Load() { return load; }

  //! Get if it is save or not.
  const bool& Save() const { return save; }

  //! Modify to be save.
  bool& Save() { return save; }

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

  //! Get if we are loading an timeseries.
  const bool& Timeseries() const { return timeseries; }

  //! Modify if we are loading an timeseries.
  bool& Timeseries() { return timeseries; }

  //! Set the sampling rate in HZ
  void SamplingRate(int hz) { samplingRate = hz; }

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

  //! Internal options. Must not be documented and used by the user.
  bool save;
  bool load;

  //! Public options.
  bool fatal;
  bool hasHeaders;
  bool noTranspose;
  bool semiColon;
  bool missingToNan;
  bool categorical;
  bool image;
  bool model;
  bool timeseries;
  int samplingRate;
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

struct HasHeadersOptions : public DataOptions
{
  inline HasHeadersOptions() : DataOptions() { this->HasHeaders() = true; }
};

struct TransposeOptions : public DataOptions
{
  inline TransposeOptions() : DataOptions() { this->NoTranspose() = false; }
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
  inline MissingToNanOptions() : DataOptions()
  {
    this->MissingToNan() = true;
  }
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
  inline AutodetectOptions() : DataOptions()
  {
    this->DataFormat() = format::autodetect;
  }
};

struct JsonDataOptions : public DataOptions
{
  inline JsonDataOptions() : DataOptions()
  {
    this->DataFormat() = format::json;
  }
};

struct XmlDataOptions : public DataOptions
{
  inline XmlDataOptions() : DataOptions()
  {
    this->DataFormat() = format::xml;
  }
};

struct BinaryDataOptions : public DataOptions
{
  inline BinaryDataOptions() : DataOptions()
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
static const DataOptionsTypes::ImageOptions         Image;
static const DataOptionsTypes::ModelOptions         Model;

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
static const DataOptionsTypes::JsonDataOptions      JSON_SER;
static const DataOptionsTypes::XmlDataOptions       XML_SER;
static const DataOptionsTypes::BinaryDataOptions    BIN_SER;


} // namespace data
} // namespace mlpack

#endif
