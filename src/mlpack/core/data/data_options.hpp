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

#include "types.hpp"
#include "dataset_mapper.hpp"
#include "map_policies/map_policies.hpp"
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
  void Reset() { }
};

using DataOptions = DataOptionsBase<EmptyOptions>;


} // namespace data
} // namespace mlpack

#endif
