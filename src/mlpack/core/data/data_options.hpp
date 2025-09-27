/**
 * @file core/data/data_options.hpp
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

#include "dataset_mapper.hpp"
#include "map_policies/map_policies.hpp"

namespace mlpack {
namespace data {

enum struct FileType
{
  FileTypeUnknown,
  AutoDetect, // attempt to automatically detect the file type
  RawASCII,   // raw text (ASCII), without a header
  ArmaASCII,  // Armadillo text format, with a header specifying matrix type and
              // size
  CSVASCII,   // comma separated values (CSV), without a header
  RawBinary,  // raw binary format (machine dependent), without a header
  ArmaBinary, // Armadillo binary format (machine dependent), with a header
              // specifying matrix type and size
  PGMBinary,  // Portable Grey Map (greyscale image)
  PPMBinary,  // Portable Pixel Map (colour image), used by the field and cube
              // classes
  HDF5Binary, // HDF5: open binary format, not specific to Armadillo, which can
              // store arbitrary data
  CoordASCII, // simple co-ordinate format for sparse matrices (indices start at
              // zero)
  ARFFASCII,  // ARFF data format, with a header specifying information about
              // categories of the data.
  JSON,       // Serialize data using Cereal library into JSON format.
  XML,        // Serialize data using Cereal library into xml format.
  BIN,        // Serialize data using Cereal library into binary format.
  ImageType,  // Any image type, STB will try to detect the type.
  PNG,        // Portable Network Graphics image type.
  JPG,        // Joint Photographic Experts Group image type.
  TGA,        // Truevision TGA image type.
  BMP,        // Bitmap image type.
  PSD,        // PhotoShop image type.
  GIF,        // Graphics Interchange Format image type.
  PIC,        // PICtor image format.
  PNM         // Portable Anymap format.
};

// This should be removed in mlpack 5.0.0. It is only here for backward
// compatibility.
enum format
{
  autodetect,
  json,
  xml,
  binary
};

/**
 * All possible DataOptions grouped under one class.
 * This will allow us to have consistent data API for mlpack. If new data
 * options might be necessary, then they should be added in the following.
 */

template<typename Derived>
class DataOptionsBase
{
 protected:
  // Users should not construct a DataOptionsBase directly.
  DataOptionsBase(const std::optional<bool> fatal = std::nullopt,
                  const std::optional<FileType> format = std::nullopt) :
      fatal(fatal),
      format(format)
  { }

 public:
  template<typename Derived2>
  explicit DataOptionsBase(const DataOptionsBase<Derived2>& opts)
  {
    CopyOptions(opts);
  }

  template<typename Derived2>
  explicit DataOptionsBase(DataOptionsBase<Derived2>&& opts)
  {
    MoveOptions(std::move(opts));
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

    CopyOptions(other);
    return *this;
  }

  // Take ownership of the options of another `DataOptionsBase` type.
  template<typename Derived2>
  DataOptionsBase& operator=(DataOptionsBase<Derived2>&& other)
  {
    if ((void*) &other != (void*) this)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    MoveOptions(std::move(other));
    return *this;
  }

  // Augment with the options of the other `DataOptionsBase`.
  template<typename Derived2>
  DataOptionsBase& operator+=(const DataOptionsBase<Derived2>& other)
  {
    Combine(other);
    return *this;
  }

  // Augment with the options of the other `DataOptionsBase`.
  template<typename Derived2>
  void Combine(const DataOptionsBase<Derived2>& other)
  {
    // Combine the fatal option.
    fatal = CombineBooleanOption(fatal, other.fatal, "Fatal()");

    // Combine the format option.
    if (format.has_value() && other.format.has_value())
    {
      // There are two cases where we can accept the other's format---when we
      // are unknown or autodetect.
      if (format == FileType::FileTypeUnknown)
      {
        // Here we always take the other format.
        format = other.format;
      }
      else if (format == FileType::AutoDetect && \
               other.format != FileType::FileTypeUnknown)
      {
        format = other.format;
      }
      else if (other.format != FileType::FileTypeUnknown && \
               other.format != FileType::AutoDetect &&
               format != other.format)
      {
        // In any other case, we won't overwrite one specified format with
        // another.
        throw std::invalid_argument("DataOptions::operator+(): cannot combine "
            "options with formats '" + FileTypeToString() + "' and '" +
            other.FileTypeToString() + "'!");
      }
    }
    else if (!format.has_value() && other.format.has_value())
    {
      // Always take the format of the other if it's unspecified.
      format = other.format;
    }

    // If the derived type is the same, we can take any options from it.
    if constexpr (std::is_same_v<Derived, Derived2>)
    {
      static_cast<Derived&>(*this).Combine(static_cast<const Derived2&>(other));
    }

    // If Derived is not the same as Derived2, we will have printed warnings in
    // the standalone operator+().
  }

  template<typename Derived2>
  void CopyOptions(const DataOptionsBase<Derived2>& other)
  {
    // Only copy options that have been set in the other object.
    if (other.fatal.has_value())
      fatal = *other.fatal;
    if (other.format.has_value())
      format = *other.format;
  }

  template<typename Derived2>
  void MoveOptions(DataOptionsBase<Derived2>&& other)
  {
    fatal = std::move(other.fatal);
    format = std::move(other.format);

    // Reset all of the options in the other object.
    other.Reset();
  }

  void Reset()
  {
    fatal.reset();
    format.reset();

    // Reset any child members.
    static_cast<Derived&>(*this).Reset();
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
   * Given a file type, return Armadillo type corresponding to that file type.
   */
  inline arma::file_type ArmaFormat() const
  {
    FileType f = format.has_value() ? *format : defaultFormat;
    switch (f)
    {
      case FileType::FileTypeUnknown:
        return arma::file_type_unknown;
        break;

      case FileType::AutoDetect:
        return arma::auto_detect;
        break;

      case FileType::RawASCII:
        return arma::raw_ascii;
        break;

      case FileType::ArmaASCII:
        return arma::arma_ascii;
        break;

      case FileType::CSVASCII:
        return arma::csv_ascii;
        break;

      case FileType::RawBinary:
        return arma::raw_binary;
        break;

      case FileType::ArmaBinary:
        return arma::arma_binary;
        break;

      case FileType::PGMBinary:
        return arma::pgm_binary;
        break;

      case FileType::PPMBinary:
        return arma::ppm_binary;
        break;

      case FileType::HDF5Binary:
        return arma::hdf5_binary;
        break;

      case FileType::CoordASCII:
        return arma::coord_ascii;
        break;

      default:
        return arma::file_type_unknown;
        break;
    }
  }

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
      case FileType::CoordASCII:
          return "ASCII formatted sparse coordinate data";
      case FileType::XML:         return "XML model";
      case FileType::BIN:         return "binary model";
      case FileType::JSON:        return "JSON model";
      case FileType::AutoDetect:  return "Detect automatically data type";
      case FileType::FileTypeUnknown: return "Unknown data type";
      case FileType::ImageType:   return "Any Image type";
      case FileType::PNG:         return "Portable Network Graphics image data";
      case FileType::JPG:         return "JPEG image data";
      case FileType::TGA:         return "Truevision TGA image data";
      case FileType::BMP:         return "Bitmap image data";
      case FileType::PSD:         return "PhotoShop image data";
      case FileType::GIF:         return "GIF image data";
      case FileType::PIC:         return "PICtor image data";
      case FileType::PNM:         return "Portable Anymap data";
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

  std::optional<bool> CombineBooleanOption(const std::optional<bool>& a,
                                           const std::optional<bool>& b,
                                           const std::string name)
  {
    if (a.has_value() && b.has_value() && ((*a) != (*b)))
    {
      // If both are set, but not the same, then throw an exception---this is
      // invalid.
      throw std::invalid_argument("DataOptions::operator+(): cannot combine "
          "options where " + name + " is set to true in one object and false "
          "in the other!");
    }
    else if (!a.has_value() && b.has_value())
    {
      // If only b is set, take b.
      return b;
    }
    else
    {
      // Otherwise, take a (whether or not it is set).
      return a;
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

// This is the class that should be used if a DataOptions with no extra options
// is meant to be constructed.
class PlainDataOptions : public DataOptionsBase<PlainDataOptions>
{
 public:
  // Allow access to all DataOptionsBase non-protected constructors and
  // operators, but with the PlainDataOptions type name.
  using DataOptionsBase::DataOptionsBase;
  using DataOptionsBase::operator=;

  // However, C++ does not allow inheriting copy and move constructors or
  // operators, and any inherited protected constructors will still be
  // protected, so forward those manually.
  PlainDataOptions(const std::optional<bool> fatal = std::nullopt,
                   const std::optional<FileType> format = std::nullopt) :
      DataOptionsBase(fatal, format) { }
  PlainDataOptions(const DataOptionsBase<PlainDataOptions>& other) :
      DataOptionsBase(other) { }
  PlainDataOptions(DataOptionsBase<PlainDataOptions>&& other) :
      DataOptionsBase(std::move(other)) { }

  PlainDataOptions& operator=(const DataOptionsBase<PlainDataOptions>& other)
  {
    return static_cast<PlainDataOptions&>(DataOptionsBase::operator=(other));
  }

  PlainDataOptions& operator=(DataOptionsBase<PlainDataOptions>&& other)
  {
    return static_cast<PlainDataOptions&>(
        DataOptionsBase::operator=(std::move(other)));
  }

  void WarnBaseConversion(const char* /* dataDescription */) const { }
  static const char* DataDescription() { return "general data"; }
  void Reset() { }
  void Combine(const PlainDataOptions&) { }
};

using DataOptions = PlainDataOptions;

// Boolean options
static const DataOptions Fatal   = DataOptions(true);
static const DataOptions NoFatal = DataOptions(false);

//! File options
static const DataOptions CSV = DataOptions(std::nullopt, FileType::CSVASCII);
static const DataOptions PGM = DataOptions(std::nullopt, FileType::PGMBinary);
static const DataOptions PPM = DataOptions(std::nullopt, FileType::PPMBinary);
static const DataOptions HDF5 = DataOptions(std::nullopt,
    FileType::HDF5Binary);
static const DataOptions ArmaAscii = DataOptions(std::nullopt,
    FileType::ArmaASCII);
static const DataOptions ArmaBin = DataOptions(std::nullopt,
    FileType::ArmaBinary);
static const DataOptions RawAscii = DataOptions(std::nullopt,
    FileType::RawASCII);
static const DataOptions BinAscii = DataOptions(std::nullopt,
    FileType::RawBinary);
static const DataOptions CoordAscii = DataOptions(std::nullopt,
    FileType::CoordASCII);
static const DataOptions AutoDetect = DataOptions(std::nullopt,
    FileType::AutoDetect);
static const DataOptions JSON = DataOptions(std::nullopt, FileType::JSON);
static const DataOptions XML = DataOptions(std::nullopt, FileType::XML);
static const DataOptions BIN = DataOptions(std::nullopt, FileType::BIN);
static const DataOptions PNG = DataOptions(std::nullopt, FileType::PNG);
static const DataOptions JPG = DataOptions(std::nullopt, FileType::JPG);
static const DataOptions TGA = DataOptions(std::nullopt, FileType::TGA);
static const DataOptions BMP = DataOptions(std::nullopt, FileType::BMP);
static const DataOptions PSD = DataOptions(std::nullopt, FileType::PSD);
static const DataOptions GIF = DataOptions(std::nullopt, FileType::GIF);
static const DataOptions PIC = DataOptions(std::nullopt, FileType::PIC);
static const DataOptions PNM = DataOptions(std::nullopt, FileType::PNM);
static const DataOptions Image = DataOptions(std::nullopt,
    FileType::ImageType);

// Utility struct to detect when something is a `DataOptions`.

template<typename T>
struct IsDataOptions
{
  constexpr static bool value = false;
};

template<typename T>
struct IsDataOptions<DataOptionsBase<T>>
{
  constexpr static bool value = true;
};

template<>
struct IsDataOptions<DataOptions>
{
  constexpr static bool value = true;
};

template<typename Derived>
bool HandleError(const std::stringstream& oss,
    const DataOptionsBase<Derived>& opts)
{
  bool success = true; // this should never be returned as true.
  if (opts.Fatal())
  {
    Log::Fatal << oss.str() << std::endl;
  }
  else
  {
    Log::Warn << oss.str() << std::endl;
    success = false;
  }
  return success;
}

template<typename Derived>
bool HandleError(const std::string& msg,
    const DataOptionsBase<Derived>& opts)
{
  std::stringstream oss;
  oss << msg;
  return HandleError(oss, opts);
}

// Required for backward compatibility.
inline bool HandleError(const std::stringstream& oss, bool fatal)
{
  bool success = true; // this should never be returned as true.
  if (fatal)
  {
    Log::Fatal << oss.str() << std::endl;
  }
  else
  {
    Log::Warn << oss.str() << std::endl;
    success = false;
  }
  return success;
}

inline bool HandleError(const std::string& msg, bool fatal)
{
  std::stringstream oss;
  oss << msg;
  return HandleError(oss, fatal);
}

} // namespace data
} // namespace mlpack

#endif
