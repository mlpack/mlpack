/**
 * @file save_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of save functionality.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
          bool fatal,
          bool transpose)
{
  Timer::Start("saving_data");

  // First we will try to discriminate by file extension.
  size_t ext = filename.rfind('.');
  if (ext == std::string::npos)
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

  // Get the actual extension.
  std::string extension = filename.substr(ext + 1);

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

}; // namespace data
}; // namespace mlpack

#endif
