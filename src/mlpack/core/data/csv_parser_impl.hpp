/* This file is originated from armadillo and adapted for mlpack
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
#ifndef MLPACK_CORE_DATA_CSV_PARSER_IMPL_HPP
#define MLPACK_CORE_DATA_CSV_PARSER_IMPL_HPP

#include "csv_parser.hpp"

namespace mlpack
{
namespace data
{

  /**
   * Given the address of a martix element(val)
   * sets it equal to the provided value(token)
   * example calling: convert_token<eT>(x.at(row, col), token)
   */
  template<typename MatType>
  bool ConvertToken(typename MatType::elem_type& val, const std::string& token)
  {
    const size_t N = size_t(token.length());

    if (N == 0)
    { 
      val = typename MatType::elem_type(0);

      return true; 
    }

    const char* str = token.c_str();

    // checking for nan, +inf, -inf
    // in both upper and lower case
    // using arma::Datum which basically
    // contains all the physical constants
    if ((N == 3) || (N == 4))
    {
      const bool neg = (str[0] == '-');
      const bool pos = (str[0] == '+');

      const size_t offset = ((neg || pos) && (N == 4)) ? 1 : 0;

      const char sig_a = str[offset];      
      const char sig_b = str[offset+1];
      const char sig_c = str[offset+2];

      if (((sig_a == 'i') || (sig_a == 'I')) && ((sig_b == 'n') || (sig_b == 'N')) 
            && ((sig_c == 'f') || (sig_c == 'F')))
      {
        // val = if(neg == true) ? -INF : +INF 
        val = neg ? -(std::numeric_limits<typename MatType::elem_type>::infinity()) : std::numeric_limits<typename MatType::elem_type>::infinity();
        
        return true;
      }
      else if (((sig_a == 'n') || (sig_a == 'N')) && ((sig_b == 'a') || (sig_b == 'A')) 
            && ((sig_c == 'n') || (sig_c == 'N')))
      {
        val = std::numeric_limits<typename MatType::elem_type>::quiet_NaN();

        return true;
      }
    }

    char* endptr = nullptr;

    if (std::is_floating_point<typename MatType::elem_type>::value || std::is_integral<typename MatType::elem_type>::value)
    {
      val = typename MatType::elem_type(std::strtod(str, &endptr));
    }
    else
    {
      if (std::is_signed<typename MatType::elem_type>::value)
      {
        val = typename MatType::elem_type(std::strtoll(str, &endptr, 10));
      }
      else
      {
        if (str[0] == '-')
        {
          val = typename MatType::elem_type(0);
          return true;
        }

        val = typename MatType::elem_type( std::strtoull(str, &endptr, 10) );
      }      
    }

    if (str == endptr)
    {
      return false;
    }

    return true;
  }

  /**
   * Loads the data from the csv file
   * into the give MatType
   */
  template<typename MatType>
  bool LoadCSVV(MatType& x, std::fstream& f, std::string&)
  {
    bool load_okay = f.good();

    f.clear();
    const std::fstream::pos_type pos1 = f.tellg();

    size_t f_n_rows = 0;
    size_t f_n_cols = 0;

    std::string line_string;
    std::stringstream line_stream;

    std::string token;

    while(f.good() && load_okay)
    {
      std::getline(f, line_string);

      if (line_string.size() == 0)
      { 
        break; 
      }

      line_stream.clear();
      line_stream.str(line_string);

      size_t line_n_cols = 0;

      while(line_stream.good())
      {
        // reading each element of the row
        std::getline(line_stream, token, ',');
        ++line_n_cols;
      }

      if (f_n_cols < line_n_cols)
      {
        f_n_cols = line_n_cols;
      }
      
      ++f_n_rows;
    }

    f.clear();
    f.seekg(pos1);

    x.zeros(f_n_rows, f_n_cols);

    size_t row = 0;

    while (f.good())
    {
      std::getline(f, line_string);

      if (line_string.size() == 0)
      {
        break;
      }

      line_stream.clear();
      line_stream.str(line_string);

      size_t col = 0;

      while (line_stream.good())
      {
        std::getline(line_stream, token, ',');

        ConvertToken<MatType>(x.at(row, col), token);

        ++col;
      }
      ++row;
    }

    return load_okay;		
  }

  template <typename MatType>
  bool LoadData(const std::string& name, MatType& x, const mlpack::data::file_type type)
  {
    // bool load_okay = false;
    std::string err_msg;
    std::string print_status;
    std::fstream f;

    f.open(name.c_str(), std::fstream::in);

    switch (type)
    {
      case mlpack::data::file_type::csv_ascii:
        return LoadCSVV<MatType>(x, f, print_status);

      case mlpack::data::file_type::file_type_unknown:
        return true;

      case mlpack::data::file_type::auto_detect:
        return true;

      case mlpack::data::file_type::raw_ascii:
        return true;

      case mlpack::data::file_type::arma_ascii:
        return true;

      case mlpack::data::file_type::raw_binary:
        return true;

      case mlpack::data::file_type::arma_binary:
        return true;

      case mlpack::data::file_type::pgm_binary:
        return true;

      case mlpack::data::file_type::ppm_binary:
        return true;

      case mlpack::data::file_type::hdf5_binary:
        return true;

      case mlpack::data::file_type::hdf5_binary_trans:
        return true;

      case mlpack::data::file_type::coord_ascii:
        return true;

        break;
    }
    return false;
  }
} // namespace data
} // namespace mlpack

#endif
