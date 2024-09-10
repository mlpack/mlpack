/**
 * @file core/data/types.hpp
 * @author Gopi M. Tatiraju
 * 
 * This file contains utilitiy functions related to types of data.
 * We have adapted all the standard types which are available in armadillo.
 * 
 * This file also contains functions to convery external file types to mlpack
 * file types. In future if we the user of mlpack needs support of an external
 * linear algebra library like armadillo, all functions related to handling the
 * types goes here.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_TYPES_HPP
#define MLPACK_CORE_DATA_TYPES_HPP

#include <iostream>
#include <limits>

namespace mlpack {
namespace data {

enum struct FileType
{
  FileTypeUnknown,
  AutoDetect, // attempt to automatically detect the file type
  RawASCII,   // raw text (ASCII), without a header
  ArmaASCII,  // Armadillo text format, with a header specifying matrix type and
              // size
  CSVASCII,   // comma separated values (CSV), without a header
  RawBinary,  // raw binary format (machine dependent), without a header
  ArmaBinary, // Armadillo binary format (machine dependent), with a header
              // specifying matrix type and size
  PGMBinary,  // Portable Grey Map (greyscale image)
  PPMBinary,  // Portable Pixel Map (colour image), used by the field and cube
              // classes
  HDF5Binary, // HDF5: open binary format, not specific to Armadillo, which can
              // store arbitrary data
  CoordASCII  // simple co-ordinate format for sparse matrices (indices start at
              // zero)
};

/**
 * This function is used to convert mlpack file types to
 * their respective Armadillo file types.
 *
 * @param type mlpack::FileType.
 */
inline arma::file_type ToArmaFileType(const FileType& type);

}  // namespace data
}  // namespace mlpack

#include "types_impl.hpp"

#endif

