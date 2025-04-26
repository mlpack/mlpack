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

  DataOptionsBase(const bool fatal = defaultFatal,
                  const FileType format = defaultFormat) :
      fatal(fatal),
      format(format)
  {
    // Do nothing.
  }

  template<typename Derived2>
  explicit DataOptionsBase(const DataOptionsBase<Derived2>& opts)
  {
    // Delegate to copy operator.
    *this = opts;
  }

  template<typename Derived2>
  explicit DataOptionsBase(DataOptionsBase<Derived2>&& opts)
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  // Convert any other DataOptions type to this DataOptions type, printing
  // warnings for any members that cannot be converted.  If this object and
  // `opts` are of the same type, then the constructor for that type will be
  // called instead.
  template<typename Derived2>
  DataOptionsBase& operator=(const DataOptionsBase<Derived2>& other)
  {
    if ((void*) &other == (void*) this)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    // Only copy options that have been set in the other object.
    if (other.fatal.has_value())
      fatal = *other.fatal;
    if (other.format.has_value())
      format = *other.format;

    return *this;
  }

  // Take ownership of the options of another `DataOptionsBase` type.
  template<typename Derived2>
  DataOptionsBase& operator=(DataOptionsBase<Derived2>&& other)
  {
    if ((void*) &other == (void*) this)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    fatal = std::move(other.fatal);
    format = std::move(other.format);

    // Reset all of the options in the other object.
    other = DataOptionsBase<Derived2>();
  }

  // If true, then exceptions are thrown on failures.
  const bool& Fatal() const { return AccessMember(fatal, defaultFatal); }
  // Modify whether or not exceptions are thrown on failures.
  bool& Fatal() { return ModifyMember(fatal, defaultFatal); }

  // Get the type of the file that will be loaded.
  const FileType& Format() const { return AccessMember(format, defaultFormat); }
  // Modify the file format to load.
  FileType& Format() { return ModifyMember(format, defaultFormat); }

  /**
   * Given a file type, return a logical name corresponding to that file type.
   */
  const std::string FileTypeToString() const
  {
    FileType f = format.has_value() ? *format : defaultFormat;
    switch (f)
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

 protected:
  template<typename T>
  const T& AccessMember(const std::optional<T>& value,
                        const T& defaultValue) const
  {
    if (value.has_value())
      return *value;
    else
      return defaultValue;
  }

  template<typename T>
  T& ModifyMember(std::optional<T>& value, const T defaultValue)
  {
    // Set the default value if needed so that (*value) has defined behavior
    // according to the spec.
    if (!value.has_value())
      value = defaultValue;

    return *value;
  }

  void WarnOptionConversion(const char* optionName, const char* dataType) const
  {
    if (fatal.has_value() && *fatal)
    {
      Log::Fatal << "Option '" << optionName << "' cannot be specified when "
          << dataType << " is being loaded!" << std::endl;
    }
    else
    {
      Log::Warn << "Option '" << optionName << "' ignored; not applicable when "
          << dataType << " is being loaded!" << std::endl;
    }
  }

 private:
  std::optional<bool> fatal;
  std::optional<FileType> format;

  constexpr static const bool defaultFatal = false;
  constexpr static const FileType defaultFormat = FileType::AutoDetect;

  // For access to internal optional members.
  template<typename Derived2>
  friend class DataOptionsBase;
};

// This utility class is meant to be used as the Derived parameter for an option
// that is not actually a derived type.  It provides the WarnBaseConversion()
// member, which does nothing.
class EmptyOptions : public DataOptionsBase<EmptyOptions>
{
 public:
  void WarnBaseConversion(const char* /* dataDescription */) const { }
  static const char* DataDescription() { return "general data"; }
};

using DataOptions = DataOptionsBase<EmptyOptions>;

template<typename Derived>
class MatrixOptionsBase : public DataOptionsBase<MatrixOptionsBase<Derived>>
{
 public:
  MatrixOptionsBase(bool noTranspose = defaultNoTranspose) :
      DataOptionsBase<MatrixOptionsBase<Derived>>(),
      noTranspose(noTranspose)
  {
    // Do Nothing.
  }

  template<typename Derived2>
  explicit MatrixOptionsBase(const MatrixOptionsBase<Derived2>& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  template<typename Derived2>
  explicit MatrixOptionsBase(MatrixOptionsBase<Derived2>&& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  // Inherit base class constructors.
  using DataOptionsBase<MatrixOptionsBase<Derived>>::DataOptionsBase;

  MatrixOptionsBase& operator=(const MatrixOptionsBase& other)
  {
    if (&other == this)
      return *this;

    if (other.noTranspose.has_value())
      noTranspose = other.noTranspose;

    return *this;
  }

  MatrixOptionsBase& operator=(MatrixOptionsBase&& other)
  {
    if (&other == this)
      return *this;

    noTranspose = std::move(other.noTranspose);

    other = MatrixOptionsBase();

    return *this;
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(const MatrixOptionsBase<Derived2>& other)
  {
    if ((void*) &other == (void*) this)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    // Only copy options that have been set in the other object.
    if (other.noTranspose.has_value())
      noTranspose = other.NoTranspose();

    // Copy base members.
    DataOptionsBase<MatrixOptionsBase<Derived>>::operator=(other);

    return *this;
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(MatrixOptionsBase<Derived2>&& other)
  {
    if ((void*) this == (void*) &other)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    noTranspose = std::move(other.noTranspose);

    // Move base members.
    DataOptionsBase<MatrixOptionsBase<Derived>>::operator=(std::move(other));

    // Reset the other object.
    other = MatrixOptionsBase<Derived2>();

    return *this;
  }

  void WarnBaseConversion(const char* dataDescription) const
  {
    if (noTranspose.has_value() && noTranspose != defaultNoTranspose)
      this->WarnOptionConversion("noTranspose", dataDescription);

    // We may potentially need to print warnings for any other converted members
    // of the derived type.
    static_cast<const Derived&>(*this).WarnBaseConversion(dataDescription);
  }

  static const char* DataDescription() { return "matrix data"; }

  // Get whether or not we will avoid transposing the matrix during load.
  bool NoTranspose() const
  {
    return this->AccessMember(noTranspose, defaultNoTranspose);
  }
  // Modify whether or not we will avoid transposing the matrix during load.
  bool& NoTranspose()
  {
    return this->ModifyMember(noTranspose, defaultNoTranspose);
  }

 private:
  std::optional<bool> noTranspose;

  constexpr static const bool defaultNoTranspose = false;

  // For access to internal optional members.
  template<typename Derived2>
  friend class MatrixOptionsBase;
};

// This utility class is meant to be used as the Derived parameter for a matrix
// option that is not actually a derived type.  It provides the
// WarnBaseConversion() member, which does nothing.
class EmptyMatrixOptions : public MatrixOptionsBase<EmptyMatrixOptions>
{
 public:
  void WarnBaseConversion(const char* /* dataDescription */) const { }
  static const char* DataDescription() { return "general data"; }
};

using MatrixOptions = MatrixOptionsBase<EmptyMatrixOptions>;

class TextOptions : public MatrixOptionsBase<TextOptions>
{
 public:
  // TODO: pass through noTranspose option?
  TextOptions(bool hasHeaders = defaultHasHeaders,
              bool semicolon = defaultSemicolon,
              bool missingToNan = defaultMissingToNan,
              bool categorical = defaultCategorical) :
      MatrixOptionsBase<TextOptions>(),
      hasHeaders(hasHeaders),
      semicolon(semicolon),
      missingToNan(missingToNan),
      categorical(categorical)
  {
    // Do Nothing.
  }

  explicit TextOptions(const TextOptions& opts) :
      MatrixOptionsBase<TextOptions>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  explicit TextOptions(TextOptions&& opts) :
      MatrixOptionsBase<TextOptions>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  // Inherit base class constructors.
  using MatrixOptionsBase<TextOptions>::MatrixOptionsBase;

  TextOptions& operator=(const TextOptions& other)
  {
    if (&other == this)
      return *this;

    if (other.hasHeaders.has_value())
      hasHeaders = *other.hasHeaders;
    if (other.semicolon.has_value())
      semicolon = *other.semicolon;
    if (other.missingToNan.has_value())
      missingToNan = *other.missingToNan;
    if (other.categorical.has_value())
      categorical = *other.categorical;

    headers = other.headers;
    datasetInfo = other.datasetInfo;

    // Copy base members.
    MatrixOptionsBase<TextOptions>::operator=(other);

    return *this;
  }

  TextOptions& operator=(TextOptions&& other)
  {
    if (&other == this)
      return *this;

    hasHeaders = std::move(other.hasHeaders);
    semicolon = std::move(other.semicolon);
    missingToNan = std::move(other.missingToNan);
    categorical = std::move(other.categorical);

    headers = std::move(other.headers);
    datasetInfo = std::move(other.datasetInfo);

    // Move base members.
    MatrixOptionsBase<TextOptions>::operator=(std::move(other));

    return *this;
  }

  // Print warnings for any members that cannot be represented by a
  // DataOptionsBase<void>.
  void WarnBaseConversion(const char* dataDescription) const
  {
    if (missingToNan.has_value() && missingToNan != defaultMissingToNan)
      this->WarnOptionConversion("missingToNan", dataDescription);
    if (semicolon.has_value() && semicolon != defaultSemicolon)
      this->WarnOptionConversion("semicolon", dataDescription);
    if (categorical.has_value() && categorical != defaultCategorical)
      this->WarnOptionConversion("categorical", dataDescription);
    if (hasHeaders.has_value() && hasHeaders != defaultHasHeaders)
      this->WarnOptionConversion("hasHeaders", dataDescription);

    // If either headers or datasetInfo are non-empty, then we take it that the
    // user has manually modified them.
    if (!headers.is_empty())
      this->WarnOptionConversion("headers", dataDescription);
    if (datasetInfo.Dimensionality() > 0)
      this->WarnOptionConversion("datasetInfo", dataDescription);
  }

  static const char* DataDescription() { return "text-file matrix data"; }

  // Get if the dataset has headers or not.
  bool HasHeaders() const
  {
    return this->AccessMember(hasHeaders, defaultHasHeaders);
  }
  // Modify if the dataset has headers.
  bool& HasHeaders()
  {
    return this->ModifyMember(hasHeaders, defaultHasHeaders);
  }

  // Get if the separator is a semicolon in the data file.
  bool Semicolon() const
  {
    return this->AccessMember(semicolon, defaultSemicolon);
  }
  // Modify the separator type in the matrix.
  bool& SemiColon()
  {
    return this->ModifyMember(semicolon, defaultSemicolon);
  }

  // Get whether missing values are converted to NaN values.
  bool MissingToNan() const
  {
    return this->AccessMember(missingToNan, defaultMissingToNan);
  }
  // Modify whether missing values are converted to NaN values.
  bool& MissingToNan()
  {
    return this->ModifyMember(missingToNan, defaultMissingToNan);
  }

  // Get whether the data should be interpreted as categorical when columns are
  // not numeric.
  bool Categorical() const
  {
    return this->AccessMember(categorical, defaultCategorical);
  }
  // Modify whether the data should be interpreted as categorical when columns
  // are not numeric.
  bool& Categorical()
  {
    return this->ModifyMember(categorical, defaultCategorical);
  }

  // Get the headers.
  const arma::field<std::string>& Headers() const { return headers; }
  // Modify the headers.
  arma::field<std::string>& Headers() { return headers; }

  // Get the DatasetInfo for categorical data.
  const data::DatasetInfo& DatasetInfo() const { return datasetInfo; }
  // Modify the DatasetInfo.
  data::DatasetInfo& DatasetInfo() { return datasetInfo; }

 private:
  std::optional<bool> hasHeaders;
  std::optional<bool> semicolon;
  std::optional<bool> missingToNan;
  std::optional<bool> categorical;

  // These are not optional, but if either is specified, then it should be taken
  // to mean that `hasHeaders` or `categorical` has been specified as true.
  arma::field<std::string> headers;
  data::DatasetInfo datasetInfo;

  constexpr static const bool defaultHasHeaders = false;
  constexpr static const bool defaultSemicolon = false;
  constexpr static const bool defaultMissingToNan = false;
  constexpr static const bool defaultCategorical = false;
};

} // namespace data
} // namespace mlpack

#endif
