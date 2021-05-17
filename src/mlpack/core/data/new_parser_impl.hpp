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
#ifndef MLPACK_CORE_DATA_NEW_PARSER_IMPL_HPP
#define MLPACK_CORE_DATA_NEW_PARSER_IMPL_HPP

#include "new_parser.hpp"

namespace mlpack
{
  namespace data
  {

    template <typename eT>
    inline bool
    convert_token(eT &val, const std::string &token)
    {
      const size_t N = size_t(token.length());

      if (N == 0)
      {
        val = eT(0);
        return true;
      }

      const char *str = token.c_str();

      if ((N == 3) || (N == 4))
      {
        const bool neg = (str[0] == '-');
        const bool pos = (str[0] == '+');

        const size_t offset = ((neg || pos) && (N == 4)) ? 1 : 0;

        const char sig_a = str[offset];
        const char sig_b = str[offset + 1];
        const char sig_c = str[offset + 2];

        if (((sig_a == 'i') || (sig_a == 'I')) && ((sig_b == 'n') || (sig_b == 'N')) && ((sig_c == 'f') || (sig_c == 'F')))
        {
          val = neg ? arma::cond_rel<arma::is_signed<eT>::value>::make_neg(arma::Datum<eT>::inf) : arma::Datum<eT>::inf;

          return true;
        }
        else if (((sig_a == 'n') || (sig_a == 'N')) && ((sig_b == 'a') || (sig_b == 'A')) && ((sig_c == 'n') || (sig_c == 'N')))
        {
          val = arma::Datum<eT>::nan;

          return true;
        }
      }

      char *endptr = nullptr;

      if (arma::is_real<eT>::value)
      {
        val = eT(std::strtod(str, &endptr));
      }
      else
      {
        if (arma::is_signed<eT>::value)
        {
          // signed integer

          val = eT(std::strtoll(str, &endptr, 10));
        }
        else
        {
          // unsigned integer

          if (str[0] == '-')
          {
            val = eT(0);
            return true;
          }

          val = eT(std::strtoull(str, &endptr, 10));
        }
      }

      if (str == endptr)
      {
        return false;
      }

      return true;
    }

    template <typename T>
    inline bool
    convert_token(std::complex<T> &val, const std::string &token)
    {
      const size_t N = size_t(token.length());
      const size_t Nm1 = N - 1;

      if (N == 0)
      {
        val = std::complex<T>(0);
        return true;
      }

      const char *str = token.c_str();

      // valid complex number formats:
      // (real,imag)
      // (real)
      // ()

      if ((token[0] != '(') || (token[Nm1] != ')'))
      {
        // no brackets, so treat the token as a non-complex number

        T val_real;

        const bool state = convert_token(val_real, token); // use the non-complex version of this function

        val = std::complex<T>(val_real);

        return state;
      }

      // does the token contain only the () brackets?
      if (N <= 2)
      {
        val = std::complex<T>(0);
        return true;
      }

      size_t comma_loc = 0;
      bool comma_found = false;

      for (size_t i = 0; i < N; ++i)
      {
        if (str[i] == ',')
        {
          comma_loc = i;
          comma_found = true;
          break;
        }
      }

      bool state = false;

      if (comma_found == false)
      {
        // only the real part is available

        const std::string token_real(&(str[1]), (Nm1 - 1));

        T val_real;

        state = convert_token(val_real, token_real); // use the non-complex version of this function

        val = std::complex<T>(val_real);
      }
      else
      {
        const std::string token_real(&(str[1]), (comma_loc - 1));
        const std::string token_imag(&(str[comma_loc + 1]), (Nm1 - 1 - comma_loc));

        T val_real;
        T val_imag;

        const bool state_real = convert_token(val_real, token_real);
        const bool state_imag = convert_token(val_imag, token_imag);

        state = (state_real && state_imag);

        val = std::complex<T>(val_real, val_imag);
      }

      return state;
    }

    //! Load a matrix in CSV text format (human readable)
    template <typename eT>
    inline bool
    load_csv_ascii(arma::Mat<eT> &x, std::istream &f, std::string &)
    {
      // TODO: replace with more efficient implementation

      bool load_okay = f.good();

      f.clear();
      const std::fstream::pos_type pos1 = f.tellg();

      //
      // work out the size

      arma::uword f_n_rows = 0;
      arma::uword f_n_cols = 0;

      std::string line_string;
      std::stringstream line_stream;

      std::string token;

      while (f.good() && load_okay)
      {
        std::getline(f, line_string);

        if (line_string.size() == 0)
        {
          break;
        }

        line_stream.clear();
        line_stream.str(line_string);

        arma::uword line_n_cols = 0;

        while (line_stream.good())
        {
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

      arma::uword row = 0;

      while (f.good())
      {
        std::getline(f, line_string);

        if (line_string.size() == 0)
        {
          break;
        }

        line_stream.clear();
        line_stream.str(line_string);

        arma::uword col = 0;

        while (line_stream.good())
        {
          std::getline(line_stream, token, ',');

          convert_token(x.at(row, col), token);

          ++col;
        }

        ++row;
      }

      return load_okay;
    }

    template <typename eT>
    inline arma_cold bool
    load_data(arma::Mat<eT> &x, const arma::file_type type, std::istream &f)
    {
      bool load_okay = false;
      std::string err_msg;
      std::string g = "y";
      switch (type)
      {
      case arma::auto_detect:
        load_okay = true;
        break;

      case arma::csv_ascii:
        load_okay = load_csv_ascii(x, f, g);
      case arma::raw_ascii:
        load_okay = true;
        break;

      case arma::arma_ascii:
        load_okay = true;
        break;

      case arma::coord_ascii:
        load_okay = true;
        break;

      case arma::raw_binary:
        load_okay = true;
        break;

      case arma::arma_binary:
        load_okay = true;
        break;

      case arma::pgm_binary:
        load_okay = true;
        break;

      case arma::hdf5_binary:
        return true;
        break;

      case arma::hdf5_binary_trans: // kept for compatibility with earlier versions of Armadillo
        return true;
        break;

      default:
        load_okay = false;
      }

      return load_okay;
    }
  } // namespace data
} // namespace mlpack

#endif
