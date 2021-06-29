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

#include "types.hpp"

namespace mlpack{
namespace data{

class Parser
{	
	public:
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
 * Parses a csv file and loads the data into a given matrix. In the first pass,
 * the function will determine the number of cols and rows in the given file.
 * Once the rows and cols are fixed we initialize the matrix with zeros. In 
 * the second pass, the function converts each value to required datatype
 * and sets it equal to val. 
 *
 * This function uses MatType as template parameter in order to provide
 * support for any type of matrices from any linear algebra library. 
 *
 * @param x Matrix in which data will be loaded
 * @param f File stream to access the data file
 */
template<typename MatType>
bool LoadCSVFile(MatType& x, std::fstream& f);

inline std::pair<int, int> GetMatSize(std::fstream& f);
};

} // namespace data
} // namespace mlpack

#include "csv_parser_impl.hpp"

#endif
