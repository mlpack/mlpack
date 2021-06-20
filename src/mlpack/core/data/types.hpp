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
 * https://gitlab.com/conradsnicta/armadillo-code/-/blob/10.5.x/include/armadillo_bits/diskio_meat.hpp
 * Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
 * Copyright 2008-2016 National ICT Australia (NICTA)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ------------------------------------------------------------------------
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

namespace mlpack
{
namespace data
{

enum struct file_type
{
  FileTypeUnknown,
  AutoDetect,        //!< attempt to automatically detect the file type
  RawASCII,          //!< raw text (ASCII), without a header
  ArmaASCII,         //!< Armadillo text format, with a header specifying matrix type and size
  CSVASCII,          //!< comma separated values (CSV), without a header
  RawBinary,         //!< raw binary format (machine dependent), without a header
  ArmaBinary,        //!< Armadillo binary format (machine dependent), with a header specifying matrix type and size
  PGMBinary,         //!< Portable Grey Map (greyscale image)
  PPMBinary,         //!< Portable Pixel Map (colour image), used by the field and cube classes
  HDF5Binary,        //!< HDF5: open binary format, not specific to Armadillo, which can store arbitrary data
  CoordASCII				 //!< simple co-ordinate format for sparse matrices (indices start at zero)
};

/**
 * WHhere should I place this fucntion?
 * This fucntion is used to convert mlpack file type to respective
 * arma file type.
 *
 * @param type Mlpack's file_type which will we converted to arma's file_type
 */
inline arma::file_type ToArmaFileType(file_type& type);

}  // namespace data
}  // namespace mlpack

#include "types_impl.hpp"

#endif

