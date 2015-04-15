/**
 * @file save_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of save functionality.
 */
#ifndef __MLPACK_CORE_DATA_SAVE_IMPL_HPP
#define __MLPACK_CORE_DATA_SAVE_IMPL_HPP

// In case it hasn't already been included.
#include "save.hpp"

namespace mlpack {
namespace data {

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
  stream.open(filename.c_str(), std::fstream::out);

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

    if (!tmp.quiet_save(stream, saveType))
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
    if (!matrix.quiet_save(stream, saveType))
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
          T& t,
          const std::string& name,
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
            << " extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << "failed.  Incorrect extension?" << std::endl;

      return false;
    }
  }

  // Open the file to save to.
  std::ofstream ofs(filename);
  if (!ofs.is_open())
  {
    if (fatal)
      Log::Fatal << "Unable to open file '" << filename << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "'." << std::endl;

    return false;
  }

  try
  {
    if (f == format::xml)
    {
      boost::archive::xml_oarchive ar(ofs);
      ar << util::CreateNVP(t, name);
    }
    else if (f == format::text)
    {
      boost::archive::text_oarchive ar(ofs);
      ar << util::CreateNVP(t, name);
    }
    else if (f == format::binary)
    {
      boost::archive::binary_oarchive ar(ofs);
      ar << util::CreateNVP(t, name);
    }
  }
  catch (boost::serialization::archive_exception& e)
  {
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

} // namespace data
} // namespace mlpack

#endif
