/* Fucntions defined in this files originate from armadillo
 * This file is originated from armadillo and adapted for mlpack
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
*/

#ifndef MLPACK_CORE_DATA_CSV_PARSER_HPP
#define MLPACK_CORE_DATA_CSV_PARSER_HPP

#include <limits>
#include <type_traits>

namespace mlpack
{
namespace data
{
  enum struct file_type : unsigned int
  {
    file_type_unknown,
    auto_detect,	//!< attempt to automatically detect the file type
    raw_ascii,		//!< raw text (ASCII), without a header
    arma_ascii,		//!< Armadillo text format, with a header specifying matrix type and size
    csv_ascii,		//!< comma separated values (CSV), without a header
    raw_binary,		//!< raw binary format (machine dependent), without a header
    arma_binary,	//!< Armadillo binary format (machine dependent), with a header specifying matrix type and size
    pgm_binary,		//!< Portable Grey Map (greyscale image)
    ppm_binary,		//!< Portable Pixel Map (colour image), used by the field and cube classes
    hdf5_binary,	//!< HDF5: open binary format, not specific to Armadillo, which can store arbitrary data
    hdf5_binary_trans, //!< [DO NOT US deprecated] as per hdf5_binary, but save/load the data with columns transposed to rows
    coord_ascii				 //!< simple co-ordinate format for sparse matrices (indices start at zero)
  };

  template<typename MatType>
  bool ConvertToken(typename MatType::elem_type& val, const std::string& token);

  template<typename MatType>
  bool LoadCSVV(MatType& x, std::fstream& f, std::string&);

 } // namespace data
} // namespace mlpack

// Include implementation
#include "csv_parser_impl.hpp"

#endif
