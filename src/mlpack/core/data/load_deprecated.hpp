/**
 * @file core/data/load_deprecated.hpp
 * @author Omar Shrit
 * @author Ryan Curtin
 *
 * Shim to the old deprecated load() functions defined in load.hpp.
 *
 * This file should be removed when releasing mlpack 5.0.0
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_DEPCRECATED_HPP
#define MLPACK_CORE_DATA_LOAD_DEPCRECATED_HPP

// In case it hasn't already been included.
#include "load.hpp"

#include "image_options.hpp"

namespace mlpack {
namespace data {

/**
 * Loads a matrix from file, guessing the filetype from the extension.  This
 * will transpose the matrix at load time (unless the transpose parameter is set
 * to false).
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - CSV (arma::csv_ascii), denoted by .csv, or optionally .txt
 *  - TSV (arma::raw_ascii), denoted by .tsv, .csv, or .txt
 *  - ASCII (arma::raw_ascii), denoted by .txt
 *  - Armadillo ASCII (arma::arma_ascii), also denoted by .txt
 *  - PGM (arma::pgm_binary), denoted by .pgm
 *  - PPM (arma::ppm_binary), denoted by .ppm
 *  - Raw binary (arma::raw_binary), denoted by .bin
 *  - Armadillo binary (arma::arma_binary), denoted by .bin
 *  - HDF5 (arma::hdf5_binary), denoted by .hdf, .hdf5, .h5, or .he5
 *
 * By default, this function will try to automatically determine the type of
 * file to load based on its extension and by inspecting the file.  If you know
 * the file type and want to specify it manually, override the default
 * `inputLoadType` parameter with the correct type above (e.g.
 * `arma::csv_ascii`.)
 *
 * If the detected file type is CSV (`arma::csv_ascii`), the first row will be
 * checked for a CSV header.  If a CSV header is not detected, the first row
 * will be treated as data; otherwise, the first row will be skipped.
 *
 * If the parameter 'fatal' is set to true, a std::runtime_error exception will
 * be thrown if the matrix does not load successfully.  The parameter
 * 'transpose' controls whether or not the matrix is transposed after loading.
 * In most cases, because data is generally stored in a row-major format and
 * mlpack requires column-major matrices, this should be left at its default
 * value of 'true'.
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, transpose the matrix after loading (default true).
 * @param inputLoadType Used to determine the type of file to load (default arma::auto_detect).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal = false,
          const bool transpose = true,
          const FileType inputLoadType = FileType::AutoDetect);

/**
 * Loads a sparse matrix from file, using arma::coord_ascii format.  This
 * will transpose the matrix at load time (unless the transpose parameter is set
 * to false).  If the filetype cannot be determined, an error will be given.
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - CSV (coord_ascii), denoted by .csv or .txt
 *  - TSV (coord_ascii), denoted by .tsv or .txt
 *  - TXT (coord_ascii), denoted by .txt
 *  - Raw binary (raw_binary), denoted by .bin
 *  - Armadillo binary (arma_binary), denoted by .bin
 *
 * If the file extension is not one of those types, an error will be given.
 * This is preferable to Armadillo's default behavior of loading an unknown
 * filetype as raw_binary, which can have very confusing effects.
 *
 * If the parameter 'fatal' is set to true, a std::runtime_error exception will
 * be thrown if the matrix does not load successfully.  The parameter
 * 'transpose' controls whether or not the matrix is transposed after loading.
 * In most cases, because data is generally stored in a row-major format and
 * mlpack requires column-major matrices, this should be left at its default
 * value of 'true'.
 *
 * @param filename Name of file to load.
 * @param matrix Sparse matrix to load contents of file into.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, transpose the matrix after loading (default true).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
          const bool fatal = false,
          const bool transpose = true,
          const FileType inputLoadType = FileType::AutoDetect);

/**
 * Load a column vector from a file, guessing the filetype from the extension.
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - CSV (csv_ascii), denoted by .csv, or optionally .txt
 *  - TSV (raw_ascii), denoted by .tsv, .csv, or .txt
 *  - ASCII (raw_ascii), denoted by .txt
 *  - Armadillo ASCII (arma_ascii), also denoted by .txt
 *  - PGM (pgm_binary), denoted by .pgm
 *  - PPM (ppm_binary), denoted by .ppm
 *  - Raw binary (raw_binary), denoted by .bin
 *  - Armadillo binary (arma_binary), denoted by .bin
 *  - HDF5, denoted by .hdf, .hdf5, .h5, or .he5
 *
 * If the file extension is not one of those types, an error will be given.
 * This is preferable to Armadillo's default behavior of loading an unknown
 * filetype as raw_binary, which can have very confusing effects.
 *
 * If the parameter 'fatal' is set to true, a std::runtime_error exception will
 * be thrown if the matrix does not load successfully.
 *
 * @param filename Name of file to load.
 * @param vec Column vector to load contents of file into.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Col<eT>& vec,
          const bool fatal = false);

/**
 * Load a row vector from a file, guessing the filetype from the extension.
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - CSV (csv_ascii), denoted by .csv, or optionally .txt
 *  - TSV (raw_ascii), denoted by .tsv, .csv, or .txt
 *  - ASCII (raw_ascii), denoted by .txt
 *  - Armadillo ASCII (arma_ascii), also denoted by .txt
 *  - PGM (pgm_binary), denoted by .pgm
 *  - PPM (ppm_binary), denoted by .ppm
 *  - Raw binary (raw_binary), denoted by .bin
 *  - Armadillo binary (arma_binary), denoted by .bin
 *  - HDF5, denoted by .hdf, .hdf5, .h5, or .he5
 *
 * If the file extension is not one of those types, an error will be given.
 * This is preferable to Armadillo's default behavior of loading an unknown
 * filetype as raw_binary, which can have very confusing effects.
 *
 * If the parameter 'fatal' is set to true, a std::runtime_error exception will
 * be thrown if the matrix does not load successfully.
 *
 * @param filename Name of file to load.
 * @param rowvec Row vector to load contents of file into.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Row<eT>& rowvec,
          const bool fatal = false);

/**
 * Loads a matrix from a file, guessing the filetype from the extension and
 * mapping categorical features with a DatasetMapper object.  This will
 * transpose the matrix (unless the transpose parameter is set to false).
 * This particular overload of Load() can only load text-based formats, such as
 * those given below:
 *
 * - CSV (csv_ascii), denoted by .csv, or optionally .txt
 * - TSV (raw_ascii), denoted by .tsv, .csv, or .txt
 * - ASCII (raw_ascii), denoted by .txt
 *
 * If the file extension is not one of those types, an error will be given.
 * This is preferable to Armadillo's default behavior of loading an unknown
 * filetype as raw_binary, which can have very confusing effects.
 *
 * If the parameter 'fatal' is set to true, a std::runtime_error exception will
 * be thrown if the matrix does not load successfully.  The parameter
 * 'transpose' controls whether or not the matrix is transposed after loading.
 * In most cases, because data is generally stored in a row-major format and
 * mlpack requires column-major matrices, this should be left at its default
 * value of 'true'.
 *
 * If the given `info` has already been used with a different `data::Load()`
 * call where the dataset has the same dimensionality, then the mappings and
 * dimension types inside of `info` will be *re-used*.  If the given `info` is a
 * new `DatasetMapper` object (e.g. its dimensionality is 0), then new mappings
 * will be created.  If the given `info` has a different dimensionality of data
 * than what is present in `filename`, an exception will be thrown.
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param info DatasetMapper object to populate with mappings and data types.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, transpose the matrix after loading.
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          DatasetInfo& info,
          const bool fatal = false,
          const bool transpose = true);

/**
 * Load a model from a file, guessing the filetype from the extension, or,
 * optionally, loading the specified format.  If automatic extension detection
 * is used and the filetype cannot be determined, an error will be given.
 *
 * The supported types of files are the same as what is supported by the
 * cereal library:
 *
 *  - json, denoted by .json
 *  - xml, denoted by .xml
 *  - binary, denoted by .bin
 *
 * The format parameter can take any of the values in the 'format' enum:
 * 'format::autodetect', 'format::json', 'format::xml', and 'format::binary'.
 * The autodetect functionality operates on the file extension (so, "file.txt"
 * would be autodetected as text).
 *
 * The name parameter should be specified to indicate the name of the structure
 * to be loaded.  This should be the same as the name that was used to save the
 * structure (otherwise, the loading procedure will fail).
 *
 * If the parameter 'fatal' is set to true, then an exception will be thrown in
 * the event of load failure.  Otherwise, the method will return false and the
 * relevant error information will be printed to Log::Warn.
 */

template<typename T>
bool Load(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal = false,
          format f = format::autodetect,
          std::enable_if_t<HasSerialize<T>::value>* = 0);

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Format() = inputLoadType;

  return Load(filename, matrix, opts);
}

// For loading data into sparse matrix
template <typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Format() = inputLoadType;

  return Load(filename, matrix, opts);
}

// For loading data into a column vector
template <typename eT>
bool Load(const std::string& filename,
          arma::Col<eT>& vec,
          const bool fatal)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  return Load(filename, vec, opts);
}

// For loading data into a raw vector
template <typename eT>
bool Load(const std::string& filename,
          arma::Row<eT>& rowvec,
          const bool fatal)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  return Load(filename, rowvec, opts);
}

// Load with mappings.  Unfortunately we have to implement this ourselves.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          DatasetInfo& info,
          const bool fatal,
          const bool transpose)
{
  TextOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Categorical() = true;
  opts.DatasetInfo() = info;

  bool success = Load(filename, matrix, opts);

  info = opts.DatasetInfo();

  return success;
}

/**
 * Image load/save interfaces.
 */

//
// Old Image loading interface, to be removed in mlpack 5.0.0
//

/**
 * Load the image file into the given matrix.
 *
 * @param filename Name of the image file.
 * @param matrix Matrix to load the image into.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& opts,
          const bool fatal)
{
  // Use the new implementation.
  opts.Fatal() = fatal;
  opts.Format() = FileType::ImageType;
  std::vector<std::string> files;
  files.push_back(filename);
  return LoadImage(files, matrix, opts);
}

/**
 * Load the image file into the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& opts,
          const bool fatal)
{
  // Use the new implementation.
  opts.Fatal() = fatal;
  opts.Format() = FileType::ImageType;
  return LoadImage(files, matrix, opts);
}

// Load a model from file.
template<typename T>
bool Load(const std::string& filename,
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
            << " extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << "failed.  Incorrect extension?" << std::endl;

      return false;
    }
  }

  // Now load the given format.
  std::ifstream ifs;
#ifdef _WIN32 // Open non-text in binary mode on Windows.
  if (f == format::binary)
    ifs.open(filename, std::ifstream::in | std::ifstream::binary);
  else
    ifs.open(filename, std::ifstream::in);
#else
  ifs.open(filename, std::ifstream::in);
#endif

  if (!ifs.is_open())
  {
    std::stringstream oss;
    oss << "Unable to open file '" << filename << "' to load object '"
        << name << "'.";
    return HandleError(oss, fatal);
  }
  try
  {
    if (f == format::xml)
    {
      cereal::XMLInputArchive ar(ifs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == format::json)
    {
     cereal::JSONInputArchive ar(ifs);
     ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == format::binary)
    {
      cereal::BinaryInputArchive ar(ifs);
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
