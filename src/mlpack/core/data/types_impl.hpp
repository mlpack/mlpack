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

#ifndef MLPACK_CORE_DATA_TYPES_IMPL_HPP
#define MLPACK_CORE_DATA_TYPES_IMPL_HPP

#include "types.hpp"

namespace mlpack{
namespace data{

inline arma::file_type ToArmaFileType(const FileType& type)
{
  switch (type)
  {
    case FileType::FileTypeUnknown:
      return arma::file_type_unknown;
      break;

    case FileType::AutoDetect:
      return arma::auto_detect;
      break;

    case FileType::RawASCII:
      return arma::raw_ascii;
      break;

    case FileType::ArmaASCII:
      return arma::arma_ascii;
      break;

    case FileType::CSVASCII:
      return arma::csv_ascii;
      break;

    case FileType::RawBinary:
      return arma::raw_binary;
      break;

    case FileType::ArmaBinary:
      return arma::arma_binary;
      break;

    case FileType::PGMBinary:
      return arma::pgm_binary;
      break;

    case FileType::PPMBinary:
      return arma::ppm_binary;
      break;

    case FileType::HDF5Binary:
      return arma::hdf5_binary;
      break;

    case FileType::CoordASCII:
      return arma::coord_ascii;
      break;

    default:
      return arma::file_type_unknown;
      break;
  }
}

}  // namespace data
}  // namespace mlpack
#endif
