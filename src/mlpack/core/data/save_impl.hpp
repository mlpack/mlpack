/**
 * @file core/data/save_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of save functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_IMPL_HPP
#define MLPACK_CORE_DATA_SAVE_IMPL_HPP

// In case it hasn't already been included.
#include "save.hpp"
#include "extension.hpp"

namespace mlpack {
namespace data {

/*
 * btw, the following two functions are not documented anywhere
 * If they are not exposed to the public API then I can delete them
 * If yes, then they are deprecated.
 * @rcurtin please comment:
 */
template<typename eT>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal,
          FileType inputSaveType)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false, inputSaveType);
}

template<typename eT>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
bool Save(const std::string& filename,
          const arma::Row<eT>& rowvec,
          const bool fatal,
          FileType inputSaveType)
{
  return Save(filename, rowvec, fatal, true, inputSaveType);
}

template<typename MatType>
bool Save(const std::string& filename,
          const MatType& matrix,
          DataOptions& opts)
{
  bool success = false;
  using eT = typename MatType::elem_type;
  if constexpr (std::is_same_v<MatType, arma::SpMat<eT>>)
  {
    success = Save(filename, matrix, opts);
  }
  else if constexpr (std::is_same_v<MatType, arma::Col<eT>>)
  {
    opts.NoTranspose() = true;
    success = Save(filename, matrix, opts);
  }
  else if constexpr (std::is_same_v<MatType, arma::Row<eT>>)
  {
    opts.NoTranspose() = false;
    success = Save(filename, matrix, opts);
  }
  else
  {
    success = Save(filename, matrix, opts);
  }

  return success;
}

// Save a Sparse Matrix
template<typename eT>
bool Save(const std::string& filename,
          const arma::SpMat<eT>& matrix,
          const bool fatal,
          bool transpose)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;

  return Save(filename, matrix, opts);
}


template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          const bool fatal,
          bool transpose,
          FileType inputSaveType)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.FileFormat() = inputSaveType;

  return Save(filename, matrix, opts);
}

//! Save a model to file.
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f)
{
  DataOptions opts;
  opts.ObjectName() = name;
  opts.Fatal() = fatal;
  opts.DataFormat() = f;

  return SaveModel(filename, t, opts);
}

template<typename MatType>
bool Save(const std::string& filename,
          MatType& matrix,
          const DataOptions& opts)
{
  // Copy the options since passing a const into a modifiable function can
  // result in a segmentation fault.
  // We do not copy back to preserve the const property of the function.
  DataOptions copyOpts = opts;
  return Save(filename, matrix, copyOpts);
}

template<typename MatType>
bool Save(const std::string& filename,
          MatType& matrix,
          DataOptions& opts)
{
  Timer::Start("saving_data");

  if (opts.FileFormat() == FileType::AutoDetect)
  {
    // Detect the file type using only the extension.
    opts.FileFormat() = DetectFromExtension(filename);
    if (opts.FileFormat() == FileType::FileTypeUnknown)
    {
      if (opts.Fatal())
        Log::Fatal << "Could not detect type of file '" << filename << "' for "
            << "writing.  Save failed." << std::endl;
      else
        Log::Warn << "Could not detect type of file '" << filename << "' for "
            << "writing.  Save failed." << std::endl;

      return false;
    }
  }

  // Catch errors opening the file.
  std::fstream stream;
#ifdef  _WIN32 // Always open in binary mode on Windows.
  stream.open(filename.c_str(), std::fstream::out | std::fstream::binary);
#else
  stream.open(filename.c_str(), std::fstream::out);
#endif
  if (!stream.is_open())
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "Cannot open file '" << filename << "' for writing. "
          << "Save failed." << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "' for writing; save "
          << "failed." << std::endl;

    return false;
  }

  // Try to save the file.
  Log::Info << "Saving " << opts.FileTypeToString() << " to '" << filename
      << "'." << std::endl;

  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
     inplace_trans(matrix);
  }

  bool success;
  if (opts.FileFormat() == FileType::HDF5Binary)
  {
#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    matrix.save(filename, ToArmaFileType(opts.FileFormat()))
#endif
  }
  else
  {
    success = matrix.save(stream, ToArmaFileType(opts.FileFormat()));
  }

  if (!success)
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "Save to '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Save to '" << filename << "' failed." << std::endl;

    return false;
  }
  

  Timer::Stop("saving_data");

  // Finally return success.
  return true;
}

// Save a Sparse Matrix
template<typename eT>
bool Save(const std::string& filename,
          arma::SpMat<eT>& matrix,
          DataOptions& opts)
{

  Timer::Start("saving_data");

  // First we will try to discriminate by file extension.
  std::string extension = Extension(filename);
  if (extension == "")
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "No extension given with filename '" << filename << "'; "
          << "type unknown.  Save failed." << std::endl;
    else
      Log::Warn << "No extension given with filename '" << filename << "'; "
          << "type unknown.  Save failed." << std::endl;

    return false;
  }

  // Catch errors opening the file.
  std::fstream stream;
#ifdef  _WIN32 // Always open in binary mode on Windows.
  stream.open(filename.c_str(), std::fstream::out | std::fstream::binary);
#else
  stream.open(filename.c_str(), std::fstream::out);
#endif
  if (!stream.is_open())
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "Cannot open file '" << filename << "' for writing. "
          << "Save failed." << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "' for writing; save "
          << "failed." << std::endl;

    return false;
  }

  bool unknownType = false;
  FileType saveType;
  std::string stringType;

  if (extension == "txt" || extension == "tsv")
  {
    saveType = FileType::CoordASCII;
    stringType = "raw ASCII formatted data";
  }
  else if (extension == "bin")
  {
    saveType = FileType::ArmaBinary;
    stringType = "Armadillo binary formatted data";
  }
  else
  {
    unknownType = true;
    saveType = FileType::RawBinary; // Won't be used; prevent a warning.
    stringType = "";
  }

  // Provide error if we don't know the type.
  if (unknownType)
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "Unable to determine format to save to from filename '"
          << filename << "'.  Save failed." << std::endl;
    else
      Log::Warn << "Unable to determine format to save to from filename '"
          << filename << "'.  Save failed." << std::endl;

    return false;
  }

  // Try to save the file.
  Log::Info << "Saving " << stringType << " to '" << filename << "'."
      << std::endl;

  arma::SpMat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = matrix;
    matrix = trans(tmp);
  }

  const bool success = matrix.save(stream, ToArmaFileType(saveType));
  if (!success)
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "Save to '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Save to '" << filename << "' failed." << std::endl;

    return false;
  }

  Timer::Stop("saving_data");

  // Finally return success.
  return true;
}

//! Save a model to file.
template<typename Object>
bool SaveModel(const std::string& filename,
               Object& objectToSerialize,
               DataOptions opts)
{
  if (opts.DataFormat() == format::autodetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      opts.DataFormat() = format::xml;
    else if (extension == "bin")
      opts.DataFormat() = format::binary;
    else if (extension == "json")
      opts.DataFormat() = format::json;
    else
    {
      if (opts.Fatal())
        Log::Fatal << "Unable to detect type of '" << filename << "'; incorrect"
            << " extension? (allowed: xml/bin/json)" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; save "
            << "failed.  Incorrect extension? (allowed: xml/bin/json)"
            << std::endl;

      return false;
    }
  }

  // Open the file to save to.
  std::ofstream ofs;
#ifdef _WIN32
  if (f == format::binary) // Open non-text types in binary mode on Windows.
    ofs.open(filename, std::ofstream::out | std::ofstream::binary);
  else
    ofs.open(filename, std::ofstream::out);
#else
  ofs.open(filename, std::ofstream::out);
#endif

  if (!ofs.is_open())
  {
    if (opts.Fatal())
      Log::Fatal << "Unable to open file '" << filename << "' to save object '"
          << opts.ObjectName() << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "' to save object '"
          << opts.ObjectName() << "'." << std::endl;

    return false;
  }

  try
  {
    if (opts.DataFormat() == format::xml)
    {
      cereal::XMLOutputArchive ar(ofs);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::json)
    {
      cereal::JSONOutputArchive ar(ofs);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::binary)
    {
      cereal::BinaryOutputArchive ar(ofs);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    return true;
  }
  catch (cereal::Exception& e)
  {
    if (opts.Fatal())
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

} // namespace data
} // namespace mlpack

#endif
