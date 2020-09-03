/**
 * @file core/data/save_impl.hpp
 * @author Ryan Curtin
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
#include "detect_file_type.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace mlpack {
namespace data {

template<typename eT>
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal,
          arma::file_type inputSaveType)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false, inputSaveType);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Row<eT>& rowvec,
          const bool fatal,
          arma::file_type inputSaveType)
{
  return Save(filename, rowvec, fatal, true, inputSaveType);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          const bool fatal,
          bool transpose,
          arma::file_type inputSaveType)
{
  Timer::Start("saving_data");

  arma::file_type saveType = inputSaveType;
  std::string stringType = "";

  if (inputSaveType == arma::auto_detect)
  {
    // Detect the file type using only the extension.
    saveType = DetectFromExtension(filename);
    if (saveType == arma::file_type_unknown)
    {
      if (fatal)
        Log::Fatal << "Could not detect type of file '" << filename << "' for "
            << "writing.  Save failed." << std::endl;
      else
        Log::Warn << "Could not detect type of file '" << filename << "' for "
            << "writing.  Save failed." << std::endl;

      return false;
    }
  }

  stringType = GetStringType(saveType);

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
    if (fatal)
      Log::Fatal << "Cannot open file '" << filename << "' for writing. "
          << "Save failed." << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "' for writing; save "
          << "failed." << std::endl;

    return false;
  }

  // Try to save the file.
  Log::Info << "Saving " << stringType << " to '" << filename << "'."
      << std::endl;

  // Transpose the matrix.
  if (transpose)
  {
    arma::Mat<eT> tmp = trans(matrix);

#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    const bool success = (saveType == arma::hdf5_binary) ?
        tmp.quiet_save(filename, saveType) :
        tmp.quiet_save(stream, saveType);
#else
    const bool success = tmp.quiet_save(stream, saveType);
#endif
    if (!success)
    {
      Timer::Stop("saving_data");
      if (fatal)
        Log::Fatal << "Save to '" << filename << "' failed." << std::endl;
      else
        Log::Warn << "Save to '" << filename << "' failed." << std::endl;

      return false;
    }
  }
  else
  {
#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    const bool success = (saveType == arma::hdf5_binary) ?
        matrix.quiet_save(filename, saveType) :
        matrix.quiet_save(stream, saveType);
#else
    const bool success = matrix.quiet_save(stream, saveType);
#endif
    if (!success)
    {
      Timer::Stop("saving_data");
      if (fatal)
        Log::Fatal << "Save to '" << filename << "' failed." << std::endl;
      else
        Log::Warn << "Save to '" << filename << "' failed." << std::endl;

      return false;
    }
  }

  Timer::Stop("saving_data");

  // Finally return success.
  return true;
}

// Save a Sparse Matrix
template<typename eT>
bool Save(const std::string& filename,
          const arma::SpMat<eT>& matrix,
          const bool fatal,
          bool transpose)
{
  Timer::Start("saving_data");

  // First we will try to discriminate by file extension.
  std::string extension = Extension(filename);
  if (extension == "")
  {
    Timer::Stop("saving_data");
    if (fatal)
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
    if (fatal)
      Log::Fatal << "Cannot open file '" << filename << "' for writing. "
          << "Save failed." << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "' for writing; save "
          << "failed." << std::endl;

    return false;
  }

  bool unknownType = false;
  arma::file_type saveType;
  std::string stringType;

  if (extension == "txt" || extension == "tsv")
  {
    saveType = arma::coord_ascii;
    stringType = "raw ASCII formatted data";
  }
  else if (extension == "bin")
  {
    saveType = arma::arma_binary;
    stringType = "Armadillo binary formatted data";
  }
  else
  {
    unknownType = true;
    saveType = arma::raw_binary; // Won't be used; prevent a warning.
    stringType = "";
  }

  // Provide error if we don't know the type.
  if (unknownType)
  {
    Timer::Stop("saving_data");
    if (fatal)
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

  arma::SpMat<eT> tmp = matrix;

  // Transpose the matrix.
  if (transpose)
  {
    tmp = trans(matrix);
  }

  const bool success = tmp.quiet_save(stream, saveType);
  if (!success)
  {
    Timer::Stop("saving_data");
    if (fatal)
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
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f)
{
  if (f == format::autodetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      f = format::xml;
    else if (extension == "bin")
      f = format::binary;
    else if (extension == "txt")
      f = format::text;
    else
    {
      if (fatal)
        Log::Fatal << "Unable to detect type of '" << filename << "'; incorrect"
            << " extension? (allowed: xml/bin/txt)" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; save "
            << "failed.  Incorrect extension? (allowed: xml/bin/txt)"
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
    if (fatal)
      Log::Fatal << "Unable to open file '" << filename << "' to save object '"
          << name << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "' to save object '"
          << name << "'." << std::endl;

    return false;
  }

  try
  {
    if (f == format::xml)
    {
      boost::archive::xml_oarchive ar(ofs);
      ar << boost::serialization::make_nvp(name.c_str(), t);
    }
    else if (f == format::text)
    {
      boost::archive::text_oarchive ar(ofs);
      ar << boost::serialization::make_nvp(name.c_str(), t);
    }
    else if (f == format::binary)
    {
      boost::archive::binary_oarchive ar(ofs);
      ar << boost::serialization::make_nvp(name.c_str(), t);
    }

    return true;
  }
  catch (boost::archive::archive_exception& e)
  {
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

/**
 * Save the given image to the given filename.
 *
 * @param filename Filename to save to.
 * @param matrix Matrix containing image to be saved.
 * @param info Information about the image (width/height/channels/etc.).
 * @param fatal Whether an exception should be thrown on save failure.
 */
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal)
{
  arma::Mat<unsigned char> tmpMatrix =
      arma::conv_to<arma::Mat<unsigned char>>::from(matrix);

  // Call out to .cpp implementation.
  return SaveImage(filename, tmpMatrix, info, fatal);
}

// Image saving API for multiple files.
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal)
{
  if (files.size() == 0)
  {
    if (fatal)
    {
      Log::Fatal << "Save(): vector of image files is empty; nothing to save."
          << std::endl;
    }
    else
    {
      Log::Warn << "Save(): vector of image files is empty; nothing to save."
          << std::endl;
    }

    return false;
  }

  arma::Mat<unsigned char> img;
  bool status = true;

  for (size_t i = 0; i < files.size() ; ++i)
  {
    arma::Mat<eT> colImg(matrix.colptr(i), matrix.n_rows, 1,
        false, true);
    status &= Save(files[i], colImg, info, fatal);
  }

  return status;
}

} // namespace data
} // namespace mlpack

#endif
