/**
 * @file save_impl.hpp
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

#include <boost/serialization/serialization.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace mlpack {
namespace data {

template<typename eT>
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Row<eT>& rowvec,
          const bool fatal)
{
  return Save(filename, rowvec, fatal, true);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
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

  if (extension == "csv")
  {
    saveType = arma::csv_ascii;
    stringType = "CSV data";
  }
  else if (extension == "txt")
  {
    saveType = arma::raw_ascii;
    stringType = "raw ASCII formatted data";
  }
  else if (extension == "bin")
  {
    saveType = arma::arma_binary;
    stringType = "Armadillo binary formatted data";
  }
  else if (extension == "pgm")
  {
    saveType = arma::pgm_binary;
    stringType = "PGM data";
  }
  else if (extension == "h5" || extension == "hdf5" || extension == "hdf" ||
           extension == "he5")
  {
#ifdef ARMA_USE_HDF5
    saveType = arma::hdf5_binary;
    stringType = "HDF5 data";
#else
    Timer::Stop("saving_data");
    if (fatal)
      Log::Fatal << "Attempted to save HDF5 data to '" << filename << "', but "
          << "Armadillo was compiled without HDF5 support.  Save failed."
          << std::endl;
    else
      Log::Warn << "Attempted to save HDF5 data to '" << filename << "', but "
          << "Armadillo was compiled without HDF5 support.  Save failed."
          << std::endl;

    return false;
#endif
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

  // Transpose the matrix.
  if (transpose)
  {
    arma::Mat<eT> tmp = trans(matrix);

    // We can't save with streams for HDF5.
    const bool success = (saveType == arma::hdf5_binary) ?
        tmp.quiet_save(filename, saveType) :
        tmp.quiet_save(stream, saveType);
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
    // We can't save with streams for HDF5.
    const bool success = (saveType == arma::hdf5_binary) ?
        matrix.quiet_save(filename, saveType) :
        matrix.quiet_save(stream, saveType);
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

#ifdef HAS_STB
// Image saving API.
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  Timer::Start("saving_image");
  // We transpose by default. So, un-transpose if necessary.
  if (!transpose)
    matrix = arma::trans(matrix);

  int tempWidth, tempHeight, tempChannels, tempQuality;

  tempWidth = info.Width();
  tempHeight = info.Height();
  tempChannels = info.Channels();
  tempQuality = info.Quality();

  if (!ImageFormatSupported(filename, true))
  {
    std::ostringstream oss;
    oss << "File type " << Extension(filename) << " not supported.\n";
    oss << "Currently it supports ";
    for (auto extension : saveFileTypes)
      oss << ", " << extension;
    oss << std::endl;
    throw std::runtime_error(oss.str());
    return false;
  }
  if (matrix.n_cols > 1)
  {
    std::cout << "Input Matrix contains more than 1 image." << std::endl;
    std::cout << "Only the firstimage will be saved!" << std::endl;
  }
  stbi_flip_vertically_on_write(transpose);

  bool status = false;
  try
  {
    unsigned char* image = matrix.memptr();

    if ("png" == Extension(filename))
    {
      status = stbi_write_png(filename.c_str(), tempWidth, tempHeight,
          tempChannels, image, tempWidth * tempChannels);
    }
    else if ("bmp" == Extension(filename))
    {
      status = stbi_write_bmp(filename.c_str(), tempWidth, tempHeight,
          tempChannels, image);
    }
    else if ("tga" == Extension(filename))
    {
      status = stbi_write_tga(filename.c_str(), tempWidth, tempHeight,
          tempChannels, image);
    }
    else if ("hdr" == Extension(filename))
    {
      status = stbi_write_hdr(filename.c_str(), tempWidth, tempHeight,
          tempChannels, reinterpret_cast<float*>(image));
    }
    else if ("jpg" == Extension(filename))
    {
      status = stbi_write_jpg(filename.c_str(), tempWidth, tempHeight,
          tempChannels, image, tempQuality);
    }
  }
  catch (std::exception& e)
  {
    Timer::Stop("saving_image");
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    Log::Warn << e.what() << std::endl;
    return false;
  }
  Timer::Stop("saving_image");
  return status;
}

// Image saving API for multiple files.
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal,
          const bool transpose)
{
  if (files.size() == 0)
  {
    std::ostringstream oss;
    oss << "Files vector is empty." << std::endl;

    throw std::runtime_error(oss.str());
    return false;
  }
  // We transpose by default. So, un-transpose if necessary.
  if (!transpose)
    matrix = arma::trans(matrix);

  arma::Mat<unsigned char> img;
  bool status = Save(files[0], img, info, fatal, transpose);

  // Decide matrix dimension using the image height and width.
  matrix.set_size(info.Width() * info.Height() * info.Channels(), files.size());
  matrix.col(0) = img;

  for (size_t i = 1; i < files.size() ; i++)
  {
    arma::Mat<unsigned char> colImg(matrix.colptr(i), matrix.n_rows, 1,
        false, true);
    status &= Save(files[i], colImg, info, fatal, transpose);
  }
  return status;
}
#else
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false,
          const bool transpose = true)
{
  throw std::runtime_error("Save(): HAS_STB is not defined, "
      "so STB is not available and images cannot be saved!");
}

template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false,
          const bool transpose = true)
{
  throw std::runtime_error("Save(): HAS_STB is not defined, "
      "so STB is not available and images cannot be saved!");
}
#endif // HAS_STB.

} // namespace data
} // namespace mlpack

#endif
