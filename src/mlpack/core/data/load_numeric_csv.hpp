/**
 * @file core/data/load_numeric_csv.hpp
 * @author Gopi Tatiraju
 *
 * Load a matrix from file. Matrix should contain only numeric data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_NUMERIC_CSV_HPP
#define MLPACK_CORE_DATA_LOAD_NUMERIC_CSV_HPP

#include "load_csv.hpp"

namespace mlpack{
namespace data{

template<typename MatType>
bool LoadCSV::ConvertToken(typename MatType::elem_type& val,
                           const std::string& token)
{
  const size_t N = size_t(token.length());
  // Fill empty data points with 0
  if (N == 0)
  {
    val = typename MatType::elem_type(0);
    return true;
  }

  const char* str = token.c_str();

  // Checks for +/-INF and NAN
  // Converts them to their equivalent representation
  // from numeric_limits 
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
      val = neg ? -(std::numeric_limits<typename MatType::elem_type>
                    ::infinity()) : std::numeric_limits<typename MatType::
                                    elem_type>::infinity();
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

  // Convert the token into ccorrect type.
  // If we have a MatType::elem_type as unsigned int,
  // it will convert all negative numbers to 0
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

template<typename MatType>
bool LoadCSV::LoadNumericCSV(MatType& x, std::fstream& f)
{
  bool load_okay = f.good();
  f.clear();
  std::pair<size_t, size_t> mat_size = GetMatrixSize<true>(f);
  x.zeros(mat_size.first, mat_size.second);
  size_t row = 0;

  std::string lineString;
  std::stringstream lineStream;
  std::string token;

  while (f.good())
  {
    // Parse the file line by line
    std::getline(f, lineString);

    if (lineString.size() == 0)
      break;

    lineStream.clear();
    lineStream.str(lineString);

    size_t col = 0;

    while (lineStream.good())
    {
      // Parse each line
      std::getline(lineStream, token, ',');

      // This will handle loading of both dense and sparse.
      // Initialize tmp_val of type MatType::elem_type with value 0.
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

inline void LoadCSV::NumericMatSize(std::stringstream& lineStream, size_t& col, const char delim)
{
  std::string token;

  while (lineStream.good())
  {
    std::getline(lineStream, token, delim);
    ++col;
  }
}

} // namespace data
} // namespace mlpack

#endif
