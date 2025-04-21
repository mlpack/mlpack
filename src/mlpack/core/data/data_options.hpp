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

template<typename Derived>
class DataOptionsBase
{
 public:

  using DerivedType = Derived;
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
  DataOptionsBase(
      bool fatal = false,
      bool noTranspose = false,
      FileType fileFormat = FileType::AutoDetect) :
    fatal(fatal),
    noTranspose(noTranspose),
    fileFormat(fileFormat)
  {
    // Do nothing.
  }

  // This function is called to convert only the base members of `opts`.
  // We are guaranteed that Derived2 != void because of the more specific 
  // overload below.
  template<typename Derived2>
  explicit DataOptionsBase(const DataOptionsBase<Derived2>& opts)
  {
    // If opts is an ImageOptions, we "lose" some things.  We want to print
    // warnings for anything that is "lost" that the user might have set.  So, 
    // we want to call Derived2::WarnBaseConversion(); however, we can only do 
    // that if Derived2 != void.
    //
    // @rcurtin This is not compiling because Derived2 is void, and this function does
    // not have WarnBaseConversion, as we have added this in CSVOptions
    static_cast<const Derived2&>(opts).WarnBaseConversion();

    // Now convert base members...
    fatal = opts.Fatal();
    noTranspose = opts.NoTranspose();
    fileFormat = opts.FileFormat();
  }

  virtual DataOptionsBase& operator=(const DataOptionsBase& other)
  {
    if (&other == this)
      return *this;

    fatal = other.fatal;
    noTranspose = other.noTranspose;
    fileFormat = other.fileFormat;
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
      case FileType::PPMBinary:   return "PGM data";
      case FileType::HDF5Binary:  return "HDF5 data";
      case FileType::CoordASCII:  return "ASCII formatted sparse coordinate data";
      case FileType::AutoDetect:  return "Detect automatically data type";
      case FileType::FileTypeUnknown: return "Unknown data type";
      default:                    return "";
    }
  }

 private:

  //! Public options.
  bool fatal;
  bool noTranspose;
  FileType fileFormat;
};

using DataOptions = DataOptionsBase<void>;

class ModelOptions : public DataOptionsBase<ModelOptions>
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

  explicit ModelOptions(const DataOptionsBase& opts) :
    DataOptionsBase(opts),
    model(false),
    dataFormat(format::binary),
    objectName("")
  {
    // Do Nothing.
  }

  virtual ModelOptions& operator=(const ModelOptions& other)
  {
    if (&other == this)
      return *this;

    model = other.model;
    dataFormat = other.dataFormat;
    objectName = other.objectName;
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

class ImageOptions : public DataOptionsBase<ImageOptions>
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

class CSVOptions : public DataOptionsBase<CSVOptions>
{

 public:

   friend DataOptions;

   CSVOptions(
       bool hasHeaders = false,
       bool semiColon = false,
       bool missingToNan = false,
       bool categorical = false,
       bool timeseries = false,
       bool timestampCol = false,
       bool labelCol = false,
       int samplingRate = 0,
       int windowSize = 0) :
     hasHeaders(hasHeaders),
     semiColon(semiColon),
     missingToNan(missingToNan),
     categorical(categorical),
     timeseries(timeseries),
     timestampCol(timestampCol),
     labelCol(labelCol),
     samplingRate(samplingRate),
     windowSize(windowSize)
  {
    // Do Nothing.
  }

  template<typename Derived>
  explicit CSVOptions(const DataOptionsBase<Derived>& opts) :
      DataOptionsBase(opts),
      hasHeaders(false),
      semiColon(false),
      missingToNan(false),
      categorical(false),
      timeseries(false),
      timestampCol(false),
      labelCol(false),
      samplingRate(0),
      windowSize(0)
  {
    // Do Nothing.
  }

  virtual CSVOptions& operator=(const CSVOptions& other)
  {
    if (&other == this)
      return *this;

    hasHeaders = other.hasHeaders;
    semiColon = other.semiColon;
    missingToNan = other.missingToNan;
    categorical = other.categorical;
    labelCol = other.labelCol;
    timestampCol = other.timestampCol;
    timeseries = other.timeseries;
    samplingRate = other.samplingRate;
    windowSize = other.windowSize;
    headers = other.headers;
    mapper = other.mapper;
  }

  // Print warnings for any members that cannot be represented by a
  // DataOptionsBase<void>.
  void WarnBaseConversion() const
  {
    // you would do this for each member that has a non-default value (my
    // warning message is not great)
    if (missingToNan)
    {
      Log::Warn << "Cannot represent missingIsNan!  Option is ignored."
          << std::endl;
    }

    if (semiColon)
    {
      Log::Warn << "Cannot represent semiColon!  Option is ignored."
          << std::endl;
    }

    if (categorical)
    {
      Log::Warn << "Cannot represent categorical!  Option is ignored."
            << std::endl;
    }

    if (hasHeaders)
    {
      Log::Warn << "Cannot represent hasHeaders!  Option is ignored."
            << std::endl;
    }

    if (labelCol)
    {
      Log::Warn << "Cannot represent labelCol!  Option is ignored."
            << std::endl;
    }

    if (timestampCol)
    {
      Log::Warn << "Cannot represent timestampCol!  Option is ignored."
            << std::endl;
    }

    if (timeseries)
    {
      Log::Warn << "Cannot represent timeseries!  Option is ignored."
            << std::endl;
    }

    if (samplingRate > 0)
    {
      Log::Warn << "Cannot represent samplingRate!  Option is ignored."
            << std::endl;
    }

    if (windowSize > 0)
    {
      Log::Warn << "Cannot represent windowSize!  Option is ignored."
            << std::endl;
    }
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

inline CSVOptions operator|(const CSVOptions& a, const CSVOptions& b)
{
  CSVOptions output;
  output.HasHeaders() = a.HasHeaders() | b.HasHeaders();
  output.SemiColon() = a.SemiColon() | b.SemiColon();
  output.MissingToNan() = a.MissingToNan() | b.MissingToNan();
  output.Categorical() = a.Categorical() | b.Categorical();
  output.Timeseries() = a.Timeseries() | b.Timeseries();
  output.TimestampCol() = a.TimestampCol() | b.TimestampCol();
  output.LabelCol() =  a.LabelCol() | b.LabelCol();
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

  if (a.FileFormat() == FileType::FileTypeUnknown ||
      a.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = b.FileFormat();
    return output;
  }
  else if (b.FileFormat() == FileType::FileTypeUnknown ||
      b.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = a.FileFormat();
    return output;
  }

  if (a.FileFormat() != b.FileFormat())
  {
    std::cout << a.FileTypeToString() << " " << b.FileTypeToString() << std::endl;
    throw std::runtime_error("File formats don't match!");
  }
  else
    output.FileFormat() = a.FileFormat();

  return output;
}

inline CSVOptions operator|(const CSVOptions& a, const DataOptions& b)
{
  CSVOptions output(a);
  output.Fatal() = a.Fatal() | b.Fatal();
  output.NoTranspose() = a.NoTranspose() | b.NoTranspose();

  if (a.FileFormat() == FileType::FileTypeUnknown ||
      a.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = b.FileFormat();
    return output;
  }
  else if (b.FileFormat() == FileType::FileTypeUnknown ||
      b.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = a.FileFormat();
    return output;
  }

  if (a.FileFormat() != b.FileFormat())
  {
    std::cout << a.FileTypeToString() << " " << b.FileTypeToString() << std::endl;
    throw std::runtime_error("File formats don't match!");
  }
  else
    output.FileFormat() = a.FileFormat();

  return output;
}

inline CSVOptions operator|(const DataOptions& a, const CSVOptions& b)
{
  CSVOptions output(b);
  output.Fatal() = a.Fatal() | b.Fatal();
  output.NoTranspose() = a.NoTranspose() | b.NoTranspose();

  if (a.FileFormat() == FileType::FileTypeUnknown ||
      a.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = b.FileFormat();
    return output;
  }
  else if (b.FileFormat() == FileType::FileTypeUnknown ||
      b.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = a.FileFormat();
    return output;
  }

  if (a.FileFormat() != b.FileFormat())
  {
    std::cout << a.FileTypeToString() << " " << b.FileTypeToString() << std::endl;
    throw std::runtime_error("File formats don't match!");
  }
  else
    output.FileFormat() = a.FileFormat();

  return output;
}

inline ModelOptions operator|(const ModelOptions& a, const DataOptions& b)
{
  ModelOptions output(a);
  std::cout << "operator model Data\n";
  output.Fatal() = a.Fatal() | b.Fatal();
  output.NoTranspose() = a.NoTranspose() | b.NoTranspose();

  if (a.FileFormat() == FileType::FileTypeUnknown ||
      a.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = b.FileFormat();
    return output;
  }
  else if (b.FileFormat() == FileType::FileTypeUnknown ||
      b.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = a.FileFormat();
    return output;
  }

  if (a.FileFormat() != b.FileFormat())
  {
    std::cout << a.FileTypeToString() << " " << b.FileTypeToString() << std::endl;
    throw std::runtime_error("File formats don't match!");
  }
  else
    output.FileFormat() = a.FileFormat();

  return output;
}

inline ModelOptions operator|(const DataOptions& a, const ModelOptions& b)
{
  ModelOptions output(b);
  std::cout << "operator data model\n";
  output.Fatal() = a.Fatal() | b.Fatal();
  output.NoTranspose() = a.NoTranspose() | b.NoTranspose();

  if (a.FileFormat() == FileType::FileTypeUnknown ||
      a.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = b.FileFormat();
    return output;
  }
  else if (b.FileFormat() == FileType::FileTypeUnknown ||
      b.FileFormat() == FileType::AutoDetect)
  {
    output.FileFormat() = a.FileFormat();
    return output;
  }

  if (a.FileFormat() != b.FileFormat())
  {
    std::cout << a.FileTypeToString() << " " << b.FileTypeToString() << std::endl;
    throw std::runtime_error("File formats don't match!");
  }
  else
    output.FileFormat() = a.FileFormat();

  return output;
}

// This will throw an error, 
//inline CSVOptions operator|(const DataOptions& a, const CSVOptions& b)
//{
  //CSVOptions output;
  //output(a | b);
  //return output;
//}

// this will throw a segmentation fault, seems that the operator = is not well
// implemented
//inline CSVOptions operator|(const DataOptions& a, const CSVOptions& b)
//{
  //CSVOptions output;
  //output = a | b;
  //return output;
//}

inline CSVOptions operator|(const CSVOptions& a, const ImageOptions& b)
{
  CSVOptions output;
  output.Fatal() = a.Fatal() | b.Fatal();
  if (output.Fatal())
    Log::Fatal << "Can't load CSV and Image option simultaneously!"
        << std::endl;
  else
    Log::Warn << "Can't load CSV and Image option simultaneously!"
        << std::endl;

  return output;
}

inline CSVOptions operator|(const CSVOptions& a, const ModelOptions& b)
{
  CSVOptions output;
  output.Fatal() = a.Fatal() | b.Fatal();
  if (output.Fatal())
    Log::Fatal << "Can't load CSV and Model option simultaneously!"
        << std::endl;
  else
    Log::Warn << "Can't load CSV and Model option simultaneously!"
        << std::endl;
  return output;
}

inline ModelOptions operator|(const ModelOptions& a, const CSVOptions& b)
{
  ModelOptions output;
  output.Fatal() = a.Fatal() | b.Fatal();
  if (output.Fatal())
    Log::Fatal << "Can't load CSV and Model option simultaneously!"
        << std::endl;
  else
    Log::Warn << "Can't load CSV and Model option simultaneously!"
        << std::endl;
  return output;
}

inline ImageOptions operator|(const ImageOptions& a, const ModelOptions& b)
{
  ImageOptions output;
  output.Fatal() = a.Fatal() | b.Fatal();
  if (output.Fatal())
    Log::Fatal << "Can't load Image and Model option simultaneously!"
        << std::endl;
  else
    Log::Warn << "Can't load Image and Model option simultaneously!"
        << std::endl;
  return output;
}

//! Boolean options
static const DataOptions Fatal       = DataOptions(true);
static const DataOptions NoFatal     = DataOptions(false);
static const DataOptions Transpose   = DataOptions(false, false);
static const DataOptions NoTranspose = DataOptions(false, true);

static const CSVOptions HasHeaders   = CSVOptions(true);
static const CSVOptions SemiColon    = CSVOptions(false, true);
static const CSVOptions MissingToNan = CSVOptions(false, false, true);
static const CSVOptions Categorical  = CSVOptions(false, false, false, true);
static const CSVOptions Timeseries   = CSVOptions(false, false, false, false,
    true);
static const CSVOptions TimestampCol = CSVOptions(false, false, false, false,
    false, true);
static const CSVOptions LabelCol     = CSVOptions(false, false, false, false,
    false, false, true);

//! File options
static const DataOptions CSV           = DataOptions(false, false,
    FileType::CSVASCII);
static const DataOptions PGM_BIN       = DataOptions(false, false,
    FileType::PGMBinary);
static const DataOptions PPM_BIN       = DataOptions(false, false,
    FileType::PPMBinary);
static const DataOptions HDF5_BIN      = DataOptions(false, false,
    FileType::HDF5Binary);
static const DataOptions ARMA_ASCII    = DataOptions(false, false,
    FileType::ArmaASCII);
static const DataOptions ARMA_BIN      = DataOptions(false, false,
    FileType::ArmaBinary);
static const DataOptions RAW_ASCII     = DataOptions(false, false,
    FileType::RawASCII);
static const DataOptions BIN_ASCII     = DataOptions(false, false,
    FileType::RawBinary);
static const DataOptions COORD_ASCII   = DataOptions(false, false,
    FileType::CoordASCII);
static const DataOptions AutoDetect_File = DataOptions(false, false,
    FileType::AutoDetect);

//! Data serialization options 
static const ModelOptions AutoDetect_SER = ModelOptions(false,
    format::autodetect);
static const ModelOptions JSON_SER= ModelOptions(false, format::json);
static const ModelOptions XML_SER = ModelOptions(false, format::xml);
static const ModelOptions BIN_SER = ModelOptions(false, format::binary);

} // namespace data
} // namespace mlpack

#endif
