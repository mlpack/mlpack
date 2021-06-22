/**
 * @file core/data/types.hpp
 * @author Gopi M. Tatiraju
 * 
 * This file contains utilitiy fucntions related to types of data.
 * We have adapted all the standard types which are available in armadillo.
 * 
 * This file also contains functions to convery external file types to mlpack
 * file types. In future if we the user of mlpack needs support of an external
 * linear algebra library like armadillo, all fucntions related to handling the
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

namespace mlpack
{
namespace data
{

inline arma::file_type ToArmaFileType(file_type& type)
{
  switch(type)
  {
    case file_type::FileTypeUnknown:
      return arma::file_type_unknown;
      break;

    case file_type::AutoDetect:
      return arma::auto_detect;
      break;
    
    case file_type::RawASCII:
      return arma::raw_ascii;
      break;
    
    case file_type::ArmaASCII:
      return arma::arma_ascii;
      break;
    
    case file_type::CSVASCII:
      return arma::csv_ascii;
      break;
    
    case file_type::RawBinary:
      return arma::raw_binary;
      break;
    
    case file_type::ArmaBinary:
      return arma::arma_binary;
      break;
    
    case file_type::PGMBinary:
      return arma::pgm_binary;
      break;
    
    case file_type::PPMBinary:
      return arma::ppm_binary;
      break;
    
    case file_type::HDF5Binary:
      return arma::hdf5_binary;
      break;
    
    case file_type::CoordASCII:
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
