/**
 * @file core/data/save_deprecated.hpp
 * @author Omar Shrit
 *
 * Contains declaration and implementation of old deprecated save function.
 * This should be removed when releasing mlpack 5.0.0.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_DEPRECATED_HPP
#define MLPACK_CORE_DATA_SAVE_DEPRECATED_HPP

// In case it hasn't already been included.
#include "save.hpp"
#include "extension.hpp"

namespace mlpack {
namespace data {

/**
 * Saves a matrix to file, guessing the filetype from the extension.  This
 * will transpose the matrix at save time.  If the filetype cannot be
 * determined, an error will be given.
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - CSV (arma::csv_ascii), denoted by .csv, or optionally .txt
 *  - ASCII (arma::raw_ascii), denoted by .txt
 *  - Armadillo ASCII (arma::arma_ascii), also denoted by .txt
 *  - PGM (arma::pgm_binary), denoted by .pgm
 *  - PPM (arma::ppm_binary), denoted by .ppm
 *  - Raw binary (arma::raw_binary), denoted by .bin
 *  - Armadillo binary (arma::arma_binary), denoted by .bin
 *  - HDF5 (arma::hdf5_binary), denoted by .hdf5, .hdf, .h5, or .he5
 *
 * By default, this function will try to automatically determine the format to
 * save with based only on the filename's extension.  If you would prefer to
 * specify a file type manually, override the default
 * `inputSaveType` parameter with the correct type above (e.g.
 * `arma::csv_ascii`.)
 *
 * If the 'fatal' parameter is set to true, a std::runtime_error exception will
 * be thrown upon failure.  If the 'transpose' parameter is set to true, the
 * matrix will be transposed before saving.  Generally, because mlpack stores
 * matrices in a column-major format and most datasets are stored on disk as
 * row-major, this parameter should be left at its default value of 'true'.
 *
 * @param filename Name of file to save to.
 * @param matrix Matrix to save into file.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, transpose the matrix before saving (default true).
 * @param inputSaveType File type to save to (defaults to arma::auto_detect).
 * @return Boolean value indicating success or failure of save.
 */
template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          const bool fatal = false,
          bool transpose = true,
          FileType inputSaveType = FileType::AutoDetect);

/**
 * Saves a sparse matrix to file, guessing the filetype from the
 * extension.  This will transpose the matrix at save time.  If the
 * filetype cannot be determined, an error will be given.
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - TSV (coord_ascii), denoted by .tsv or .txt
 *  - TXT (coord_ascii), denoted by .txt
 *  - Raw binary (raw_binary), denoted by .bin
 *  - Armadillo binary (arma_binary), denoted by .bin
 *
 * If the file extension is not one of those types, an error will be given.  If
 * the 'fatal' parameter is set to true, a std::runtime_error exception will be
 * thrown upon failure.  If the 'transpose' parameter is set to true, the matrix
 * will be transposed before saving.  Generally, because mlpack stores matrices
 * in a column-major format and most datasets are stored on disk as row-major,
 * this parameter should be left at its default value of 'true'.
 *
 * @param filename Name of file to save to.
 * @param matrix Sparse matrix to save into file.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, transpose the matrix before saving (default true).
 * @return Boolean value indicating success or failure of save.
 */
template<typename eT>
bool Save(const std::string& filename,
          const arma::SpMat<eT>& matrix,
          const bool fatal = false,
          bool transpose = true);

/**
 * Saves a model to file, guessing the filetype from the extension, or,
 * optionally, saving the specified format.  If automatic extension detection is
 * used and the filetype cannot be determined, and error will be given.
 *
 * The supported types of files are the same as what is supported by the
 * cereal library:
 *
 *  - JSON, denoted by .json
 *  - XML, denoted by .xml
 *  - BIN, denoted by .bin
 *
 * The FileType parameter can take any of the model-specific values in the
 * 'FileType' enum: 'FileType::Autodetect', 'FileType::JSON', 'FileType::XML',
 * and 'FileType::BIN'. The autodetect functionality operates on the file
 * extension (so, "file.txt" would be autodetected as text).
 *
 * The name parameter should be specified to indicate the name of the structure
 * to be saved.  If Load() is later called on the generated file, the name used
 * to load should be the same as the name used for this call to Save().
 *
 * If the parameter 'fatal' is set to true, then an exception will be thrown in
 * the event of a save failure.  Otherwise, the method will return false and the
 * relevant error information will be printed to Log::Warn.
 */
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal = false,
          format f = format::autodetect,
          std::enable_if_t<HasSerialize<T>::value>* = 0);

template<typename eT>
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal,
          FileType inputSaveType)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false, inputSaveType);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Row<eT>& rowvec,
          const bool fatal,
          FileType inputSaveType)
{
  return Save(filename, rowvec, fatal, true, inputSaveType);
}

// Save a Sparse Matrix
template<typename eT>
bool Save(const std::string& filename,
          const arma::SpMat<eT>& matrix,
          const bool fatal,
          bool transpose)
{
  MatrixOptions opts;
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
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Format() = inputSaveType;

  return Save(filename, matrix, opts);
}

/**
 * Save the image file from the given matrix.
 *
 * @param filename Name of the image file.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          ImageInfo& opts,
          const bool fatal)
{
  opts.Fatal() = fatal;
  opts.Format() = FileType::ImageType;
  std::vector<std::string> files;
  files.push_back(filename);
  return SaveImage(files, matrix, opts);
}

/**
 * Save the image file from the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Save(const std::vector<std::string>& files,
          const arma::Mat<eT>& matrix,
          ImageInfo& opts,
          const bool fatal)
{
  opts.Fatal() = fatal;
  opts.Format() = FileType::ImageType;
  return SaveImage(files, matrix, opts);
}

// Save a model to file.
// Keep this implementation until mlpack 5.0.0 Then we can remove it.
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f,
          std::enable_if_t<HasSerialize<T>::value>*)
{
  if (f == format::autodetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      f = format::xml;
    else if (extension == "bin")
      f = format::binary;
    else if (extension == "json")
      f = format::json;
    else
    {
      if (fatal)
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
      cereal::XMLOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == format::json)
    {
      cereal::JSONOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == format::binary)
    {
      cereal::BinaryOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }

    return true;
  }
  catch (cereal::Exception& e)
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
