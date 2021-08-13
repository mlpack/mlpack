/** 
 * @file core/data/load_csv_impl.hpp
 * 
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
#ifndef MLPACK_CORE_DATA_CSV_PARSER_IMPL_HPP
#define MLPACK_CORE_DATA_CSV_PARSER_IMPL_HPP

#include "load_csv.hpp"

namespace mlpack
{
namespace data
{

/**
* Given the address of a matrix element(val)
* sets it equal to the provided value(token)
* example calling: convert_token(x.at(row, col), token)
*/
template<typename MatType>
bool LoadCSV::ConvertToken(typename MatType::elem_type& val,
                           const std::string& token)
{
  const size_t N = size_t(token.length());

  if (N == 0)
  {
    val = typename MatType::elem_type(0);
    return true;
  }

  const char* str = token.c_str();

  if ((N == 3) || (N == 4))
  {
    const bool neg = (str[0] == '-');
    const bool pos = (str[0] == '+');

    const size_t offset = ((neg || pos) && (N == 4)) ? 1 : 0;

    const char sig_a = str[offset];
    const char sig_b = str[offset+1];
    const char sig_c = str[offset+2];

    if (((sig_a == 'i') || (sig_a == 'I')) &&
        ((sig_b == 'n') || (sig_b == 'N')) &&
        ((sig_c == 'f') || (sig_c == 'F')))
    {
      val = neg ? -(std::numeric_limits<typename MatType::elem_type>::infinity()) :
      std::numeric_limits<typename MatType::elem_type>::infinity();
      return true;
    }
    else if (((sig_a == 'n') || (sig_a == 'N')) &&
             ((sig_b == 'a') || (sig_b == 'A')) &&
             ((sig_c == 'n') || (sig_c == 'N')))
    {
      val = std::numeric_limits<typename MatType::elem_type>::quiet_NaN();
      return true;
    }
  }

  char* endptr = nullptr;

  if (std::is_floating_point<typename MatType::elem_type>::value)
  {
    val = typename MatType::elem_type(std::strtod(str, &endptr));
  }
  else if (std::is_integral<typename MatType::elem_type>::value)
  {
    if (std::is_signed<typename MatType::elem_type>::value)
      val = typename MatType::elem_type(std::strtoll(str, &endptr, 10));
    else
    {
      if (str[0] == '-')
      {
        val = typename MatType::elem_type(0);
        return true;
      }

      val = typename MatType::elem_type( std::strtoull(str, &endptr, 10));
    }
  }

  if (str == endptr)
    return false;

  return true;
}

inline void LoadCSV::NumericParse(std::stringstream& lineStream, size_t& col, const char delim)
{
  std::string token;

  while (lineStream.good())
  {
    std::getline(lineStream, token, delim);
    ++col;
  }
}

inline void LoadCSV::CategoricalParse(std::stringstream& lineStream, size_t& col, const char delim)
{
  std::string token;

  while (lineStream.good())
  {
    std::getline(lineStream, token, delim);

    if (token[0] == '"' && token[token.size() - 1] != '"')
    {
      while (token[token.size() - 1] != '"')
        std::getline(lineStream, token, delim);
    }

    ++col;
  }
}

inline std::pair<size_t, size_t> LoadCSV::GetMatSize(std::fstream& f, const bool isNumeric = true, const char delim = ',')
{
  bool load_okay = f.good();

  f.clear();

  const std::fstream::pos_type pos1 = f.tellg();

  size_t f_n_rows = 0;
  size_t f_n_cols = 0;

  std::string lineString;
  std::stringstream lineStream;
  std::string token;

  while (f.good() && load_okay)
  {
    std::getline(f, lineString);
    if (lineString.size() == 0)
      break;

    lineStream.clear();
    lineStream.str(lineString);

    size_t line_n_cols = 0;

    if (isNumeric)
      NumericParse(lineStream, line_n_cols, delim);
    else
      CategoricalParse(lineStream, line_n_cols, delim);

    if (f_n_cols < line_n_cols)
      f_n_cols = line_n_cols;

    ++f_n_rows;
  }

  f.clear();
  f.seekg(pos1);

  std::pair<size_t, size_t> mat_size(f_n_rows, f_n_cols);

  return mat_size;
}

/**
* Returns a bool value showing whether data was loaded successfully or not.
* Parses the file and loads the data into the given matrix.
*/
template<typename MatType>
bool LoadCSV::LoadCSVFile(MatType& x, std::fstream& f)
{
  bool load_okay = f.good();

  f.clear();

  std::pair<size_t, size_t> mat_size = GetMatSize(f);

  x.zeros(mat_size.first, mat_size.second);

  size_t row = 0;

  std::string lineString;
  std::stringstream lineStream;
  std::string token;

  while (f.good())
  {
    std::getline(f, lineString);

    if (lineString.size() == 0)
      break;

    lineStream.clear();
    lineStream.str(lineString);

    size_t col = 0;

    while (lineStream.good())
    {
      std::getline(lineStream, token, ',');
      typename MatType::elem_type tmp_val = typename MatType::elem_type(0);

      if (ConvertToken<MatType>(tmp_val, token))
      {
        x.at(row, col) = tmp_val;
        ++col;
      }
    }

    ++row;
  }
  return load_okay;
}

} // namespace data
} // namespace mlpack

#endif
