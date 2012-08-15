/**
 * @file save.hpp
 * @author Ryan Curtin
 *
 * Save an Armadillo matrix to file.  This is necessary because Armadillo does
 * not transpose matrices upon saving, and it allows us to give better error
 * output.
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_DATA_SAVE_HPP
#define __MLPACK_CORE_DATA_SAVE_HPP

#include <mlpack/core/util/log.hpp>
#include <mlpack/core/arma_extend/arma_extend.hpp> // Includes Armadillo.
#include <string>

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
 *
 * If the file extension is not one of those types, an error will be given.  If
 * the 'fatal' parameter is set to true, an error will cause the program to
 * exit.  If the 'transpose' parameter is set to true, the matrix will be
 * transposed before saving.  Generally, because MLPACK stores matrices in a
 * column-major format and most datasets are stored on disk as row-major, this
 * parameter should be left at its default value of 'true'.
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
          bool fatal = false,
          bool transpose = true);

}; // namespace data
}; // namespace mlpack

// Include implementation.
#include "save_impl.hpp"

#endif
