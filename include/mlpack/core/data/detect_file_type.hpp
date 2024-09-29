/**
 * @file core/data/detect_file_type.hpp
 * @author Conrad Sanderson
 * @author Ryan Curtin
 *
 * Functionality to guess the type of a file by inspecting it.  Parts of the
 * implementation are adapted from the Armadillo sources and relicensed to be a
 * part of mlpack with permission from Conrad.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DETECT_FILE_TYPE_HPP
#define MLPACK_CORE_DATA_DETECT_FILE_TYPE_HPP

#include "types.hpp"

namespace mlpack {
namespace data {

/**
 * Given a file type, return a logical name corresponding to that file type.
 *
 * @param type Type to get the logical name of.
 */
inline std::string GetStringType(const FileType& type);

/**
 * Given an istream, attempt to guess the file type.  This is taken originally
 * from Armadillo's function guess_file_type_internal(), but we avoid using
 * internal Armadillo functionality.
 *
 * If the file is detected as a CSV, and the CSV is detected to have a header
 * row, the stream `f` will be fast-forwarded to point at the second line of the
 * file.
 *
 * @param f Opened istream to look into to guess the file type.
 */
inline FileType GuessFileType(std::istream& f);

/**
 * Attempt to auto-detect the type of a file given its extension, and by
 * inspecting the parts of the file to disambiguate between types when
 * necessary.  (For instance, a .csv file could be delimited by spaces, commas,
 * or tabs.)  This is meant to be used during loading.
 *
 * If the file is detected as a CSV, and the CSV is detected to have a header
 * row, `stream` will be fast-forwarded to point at the second line of the file.
 *
 * @param stream Opened file stream to look into for autodetection.
 * @param filename Name of the file.
 * @return The detected file type.  arma::file_type_unknown if unknown.
 */
inline FileType AutoDetect(std::fstream& stream,
                           const std::string& filename);

/**
 * Return the type based only on the extension.
 *
 * @param filename Name of the file whose type we should detect.
 * @return Detected type of file.  arma::file_type_unknown if unknown.
 */
inline FileType DetectFromExtension(const std::string& filename);

/**
 * Count the number of columns in the file.  The file must be a CSV/TSV/TXT file
 * with no header.
 */
inline size_t CountCols(std::fstream& stream);

} // namespace data
} // namespace mlpack

#include "detect_file_type_impl.hpp"

#endif
