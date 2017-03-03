/**
 * @file save.hpp
 * @author Ryan Curtin
 *
 * Save an Armadillo matrix to file.  This is necessary because Armadillo does
 * not transpose matrices upon saving, and it allows us to give better error
 * output.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_HPP
#define MLPACK_CORE_DATA_SAVE_HPP

#include <mlpack/core/util/log.hpp>
#include <mlpack/core/arma_extend/arma_extend.hpp> // Includes Armadillo.
#include <string>

#include "format.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices. */ {

/**
 * Saves a matrix to file, guessing the filetype from the extension.  This
 * will transpose the matrix at save time.  If the filetype cannot be
 * determined, an error will be given.
 *
 * The supported types of files are the same as found in Armadillo:
 *
 *  - CSV (csv_ascii), denoted by .csv, or optionally .txt
 *  - ASCII (raw_ascii), denoted by .txt
 *  - Armadillo ASCII (arma_ascii), also denoted by .txt
 *  - PGM (pgm_binary), denoted by .pgm
 *  - PPM (ppm_binary), denoted by .ppm
 *  - Raw binary (raw_binary), denoted by .bin
 *  - Armadillo binary (arma_binary), denoted by .bin
 *  - HDF5 (hdf5_binary), denoted by .hdf5, .hdf, .h5, or .he5
 *
 * If the file extension is not one of those types, an error will be given.  If
 * the 'fatal' parameter is set to true, a std::runtime_error exception will be
 * thrown upon failure.  If the 'transpose' parameter is set to true, the matrix
 * will be transposed before saving.  Generally, because mlpack stores matrices
 * in a column-major format and most datasets are stored on disk as row-major,
 * this parameter should be left at its default value of 'true'.
 *
 * @param filename Name of file to save to.
 * @param matrix Matrix to save into file.
 * @param fatal If an error should be reported as fatal (default false).
 * @param transpose If true, transpose the matrix before saving.
 * @return Boolean value indicating success or failure of save.
 */
template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          const bool fatal = false,
          bool transpose = true);

/**
 * Saves a model to file, guessing the filetype from the extension, or,
 * optionally, saving the specified format.  If automatic extension detection is
 * used and the filetype cannot be determined, and error will be given.
 *
 * The supported types of files are the same as what is supported by the
 * boost::serialization library:
 *
 *  - text, denoted by .txt
 *  - xml, denoted by .xml
 *  - binary, denoted by .bin
 *
 * The format parameter can take any of the values in the 'format' enum:
 * 'format::autodetect', 'format::text', 'format::xml', and 'format::binary'.
 * The autodetect functionality operates on the file extension (so, "file.txt"
 * would be autodetected as text).
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
          format f = format::autodetect);

} // namespace data
} // namespace mlpack

// Include implementation.
#include "save_impl.hpp"

#endif
