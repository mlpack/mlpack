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

#ifndef MLPACK_CORE_DATA_NEW_PARSER_HPP
#define MLPACK_CORE_DATA_NEW_PARSER_HPP

namespace mlpack {
namespace data {

template<typename eT>
inline
bool
convert_token(eT& val, const std::string& token);

template<typename T>
inline
bool
convert_token(std::complex<T>& val, const std::string& token);

template<typename eT>
inline
bool
load_csv_ascii(arma::Mat<eT>& x, std::istream& f, std::string&);

template<typename eT>
inline
arma_cold
bool
load_data(const std::string name, const arma::file_type type);


    }   // namespace data
}	// namespace mlpack

// Include implementation
#include "new_parser_impl.hpp"

#endif
