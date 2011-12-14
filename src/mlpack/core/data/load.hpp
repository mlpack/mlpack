/**
 * @file load.hpp
 * @author Ryan Curtin
 *
 * Load an Armadillo matrix from file.  This is necessary because Armadillo does
 * not transpose matrices on input, and it allows us to give better error
 * output.
 */
#ifndef __MLPACK_CORE_DATA_LOAD_HPP
#define __MLPACK_CORE_DATA_LOAD_HPP

#include <mlpack/core/util/log.hpp>
#include <mlpack/core/arma_extend/arma_extend.hpp> // Includes Armadillo.
#include <string>

namespace mlpack {
namespace data /** Functions to load and save matrices. */ {

/**
 * Loads a matrix from file, guessing the filetype from the extension.  This
 * will transpose the matrix at load time.  If the filetype cannot be
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
 *
 * If the file extension is not one of those types, an error will be given.
 * This is preferable to Armadillo's default behavior of loading an unknown
 * filetype as raw_binary, which can have very confusing effects.
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          bool fatal = false);

}; // namespace data
}; // namespace mlpack

// Include implementation.
#include "load_impl.hpp"

#endif
