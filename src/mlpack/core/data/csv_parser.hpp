/** 
 * @file core/data/csv_parser.hpp
 * @author Conrad Sanderson
 * @author Gopi M. Tatiraju
 *
 * This csv parser is designed by taking reference from armadillo's csv parser.
 * In this mlpack's version, all the arma dependencies were removed or replaced
 * accordingly, making the parser totally independent of armadillo.
 *
 * This parser will be totally independent to any linear algebra library.
 * This can be used to load data into any matrix, i.e. arma and bandicoot
 * in future.
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
#ifndef MLPACK_CORE_DATA_CSV_PARSER_HPP
#define MLPACK_CORE_DATA_CSV_PARSER_HPP

#include <limits>

namespace mlpack{
namespace data{

enum struct file_type : unsigned int
{
  mlp_file_type_unknown,
  mlp_auto_detect,        //!< attempt to automatically detect the file type
  mlp_raw_ascii,          //!< raw text (ASCII), without a header
  mlp_arma_ascii,         //!< Armadillo text format, with a header specifying matrix type and size
  mlp_csv_ascii,          //!< comma separated values (CSV), without a header
  mlp_raw_binary,         //!< raw binary format (machine dependent), without a header
  mlp_arma_binary,        //!< Armadillo binary format (machine dependent), with a header specifying matrix type and size
  mlp_pgm_binary,         //!< Portable Grey Map (greyscale image)
  mlp_ppm_binary,         //!< Portable Pixel Map (colour image), used by the field and cube classes
  mlp_hdf5_binary,        //!< HDF5: open binary format, not specific to Armadillo, which can store arbitrary data
  mlp_hdf5_binary_trans,  //!< [DO NOT US deprecated] as per hdf5_binary, but save/load the data with columns transposed to rows
  mlp_coord_ascii         //!< simple co-ordinate format for sparse matrices (indices start at zero)
  };

/**
 * WHere should I place this fucntion?
 * This fucntion is used to convert mlpack file type to respective
 * arma file type.
 *
 * @param type Mlpack's file_type which will we converted to arma's file_type
 */
arma::file_type ToArmaFileType(file_type& type);

/**
 * Convert the given string token to assigned datatype and assign
 * this value to the given address. The address here will be a
 * matrix location.
 * 
 * Token is always read as a string, if the given token is +/-INF or NAN
 * it converts them to infinity and NAN using numeric_limits.
 *
 * @param val Token's value will be assigned to this address
 * @param token Value which should be assigned
 */
template<typename MatType>
bool ConvertToken(typename MatType::elem_type& val, const std::string& token);

/**
 * Returns a bool value showing whether data was loaded successfully or not.
 *
 * Parses the file and loads the data into the given matrix. It will make the
 * first parse to determine the number of cols and rows in the given file.
 * Once the rows and cols are fixed we initialize a matrix of size(which we
 * calculated in the first pass) and fill it with zeros. In the second pass
 * it converts each value to required datatype and sets it equal to val.
 * 
 * Using MatType as template parameter here so that in future if mlpack
 * decides to use any other linear algebra library or want to support
 * multiple linear algebra libraries, we can make the transition easily.
 * This is to make the csv parser as generic as possible.
 * 
 * @param x Matrix in which data will be loaded
 * @param f File stream to access the data file
 */
template<typename MatType>
bool LoadCSVV(MatType& x, std::fstream& f);

} // namespace data
} // namespace mlpack

#endif
