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
  template<typename eT>
  inline bool convert_token(eT& val, const std::string& token)
  {
    const size_t N = size_t(token.length());

    if (N == 0)
    { 
      val = eT(0); 
      return true; 
    }

    const char* str = token.c_str();

    if ((N == 3) || (N == 4))
    {
      const bool neg = (str[0] == '-');
      const bool pos = (str[0] == '+');

      const size_t offset = ((neg || pos) && (N == 4)) ? 1 : 0;

      // discuss about this fucntion
    }

    char* endptr = nullptr;

    if (is_real<eT>::value)
    {
      val = eT(std::strtod(str, &endptr));
    }

    if(str == endptr) { return false; }

    return true;
  }

  template<typename eT>
  inline bool load_csv_ascii(arma::Mat<eT>& x, std::istream& f, std::string&)
  {
    bool load_okay = f.good();

    f.clear();
    const std::fstream::pos_type pos1 = f.tellg();

    // use own implementation as you don't
    // wanna depend on arma in core mlpack
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

        convert_token<eT>(x.at(row, col), token);

        ++col;
      }
      ++row;
    }

    return load_okay;		
  }

  template<typename eT>
  inline bool LoadCSV(const std::string& name, std::string& err_msg)
  {
    std::fstream f;
    arma::Mat<eT> x;

    f.open(name.c_str(), std::fstream::in);

    bool load_okay = f.is_open();

    if (load_okay == false)
    {
      return false;
    }

    if (load_okay)
    {
      load_okay = LoadCSV<eT>(x, f, err_msg);
    }

    f.close();

    return load_okay;
  }

  template<typename eT>
  inline
  bool
  Load(const std::string& name, const file_type type, const bool print_status)
  {
    bool load_okay = false;
    std::string err_msg;

    load_okay = LoadCSV<eT>(name, err_msg);

    return load_okay;
  }

  template <typename eT>
  inline bool LoadData(const std::string& name, const file_type type, const bool print_status)
  {
    bool load_okay = false;
    std::string err_msg;

    switch (type)
    {
      case file_type::csv_ascii:
        return Load<eT>(name, type, print_status);
        
        // For own implementation
        // return load(name, type, print_status);
        
        break;
    }
  }
} // namespace data
} // namespace mlpack

#endif
