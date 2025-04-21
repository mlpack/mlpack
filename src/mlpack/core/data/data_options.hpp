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

  DataOptionsBase(
      bool fatal = false,
      FileType fileFormat = FileType::AutoDetect) :
    fatal(fatal),
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
    static_cast<const Derived2&>(opts).WarnBaseConversion();

    // Now convert base members...
    fatal = opts.Fatal();
    fileFormat = opts.FileFormat();
  }

  DataOptionsBase& operator=(const DataOptionsBase& other)
  {
    if (&other == this)
      return *this;

    fatal = other.fatal;
    fileFormat = other.fileFormat;
    return *this;
  }

  // These are accessible to users.
  //! Get the error it it is fatal or not.
  const bool& Fatal() const { return fatal; }

  //! Modify the error to be fatal.
  bool& Fatal() { return fatal; }

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

  bool fatal;
  FileType fileFormat;
};

using DataOptions = DataOptionsBase<void>;

template<typename Derived>
class MatrixOptionsBase : public DataOptionsBase<MatrixOptionsBase<Derived>>
{

 public:

  MatrixOptionsBase(bool noTranspose = false):
      noTranspose(noTranspose)
  {
    // Do Nothing.
  }

  template<typename Derived2>
  explicit MatrixOptionsBase(const DataOptionsBase<Derived2>& opts) :
      DataOptionsBase(opts),
      noTranspose(noTranspose)
  {
    // Do Nothing.
  }

  MatrixOptionsBase& operator=(const MatrixOptionsBase& other)
  {
    if (&other == this)
      return *this;

    noTranspose = other.noTranspose;
    return *this;
  }
 
  // Use SFINAE to warn about members that can't be converted only in MatrixOptions.
  void WarnBaseConversion(const std::enable_if_t<std::is_same_v<Derived, void>>* = 0)
  {
    if (noTranspose)
    {
      Log::Warn << "Cannot represent noTranspose!  Option is ignored."
          << std::endl;
    }
  }

  void WarnBaseConversion(const std::enable_if_t<!std::is_same_v<Derived, void>>* = 0)
  {
    if (noTranspose)
    {
      Log::Warn << "Cannot represent noTranspose!  Option is ignored."
          << std::endl;
    }

    // this is not a MatrixOptions only, so we need to cast to the true type and call WarnBaseConversion
    static_cast<const Derived&>(*this).WarnBaseConversion();
  }

  //! Get if the matrix is transposed or not.
  const bool& NoTranspose() const { return noTranspose; }

  //! Transpose the matrix if necessary.
  bool& NoTranspose() { return noTranspose; }

 private:

  bool noTranspose;
};

using MatrixOptions = MatrixOptionsBase<void>;

class TextOptions : public MatrixOptionsBase<TextOptions>
{

 public:

   TextOptions(
       bool hasHeaders = false,
       bool semiColon = false,
       bool missingToNan = false,
       bool categorical = false) :
     hasHeaders(hasHeaders),
     semiColon(semiColon),
     missingToNan(missingToNan),
     categorical(categorical)
  {
    // Do Nothing.
  }

  template<typename Derived>
  explicit TextOptions(const MatrixOptionsBase<Derived>& opts) :
      MatrixOptionsBase(opts),
      hasHeaders(false),
      semiColon(false),
      missingToNan(false),
      categorical(false),
  {
    // Do Nothing.
  }

  TextOptions& operator=(const TextOptions& other)
  {
    if (&other == this)
      return *this;

    hasHeaders = other.hasHeaders;
    semiColon = other.semiColon;
    missingToNan = other.missingToNan;
    categorical = other.categorical;
    headers = other.headers;
    mapper = other.mapper;
    return *this;
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
  arma::field<std::string> headers;
  DatasetInfo mapper;
};

} // namespace data
} // namespace mlpack

#endif
