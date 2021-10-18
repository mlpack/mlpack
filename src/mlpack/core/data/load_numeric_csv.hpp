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

template<typename eT>
bool LoadCSV::ConvertToken(eT& val,
                           const std::string& token)
{
  const size_t N = size_t(token.length());
  // Fill empty data points with 0
  if (N == 0)
  {
    val = eT(0);
    return true;
  }

  const char* str = token.c_str();

  // Checks for +/-INF and NAN
  // Converts them to their equivalent representation
  // from numeric_limits. 
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
      val = neg ? -(std::numeric_limits<eT>
                    ::infinity()) : std::numeric_limits<eT>::infinity();
      return true;
    }
    else if (((sig_a == 'n') || (sig_a == 'N')) &&
             ((sig_b == 'a') || (sig_b == 'A')) &&
             ((sig_c == 'n') || (sig_c == 'N')))
    {
      val = std::numeric_limits<eT>::quiet_NaN();
      return true;
    }
  }

  char* endptr = nullptr;

  // Convert the token into correct type.
  // If we have a eT as unsigned int,
  // it will convert all negative numbers to 0.
  if (std::is_floating_point<eT>::value)
  {
    val = eT(std::strtod(str, &endptr));
  }
  else if (std::is_integral<eT>::value)
  {
    if (std::is_signed<eT>::value)
      val = eT(std::strtoll(str, &endptr, 10));
    else
    {
      if (str[0] == '-')
      {
        val = eT(0);
        return true;
      }
      val = eT(std::strtoull(str, &endptr, 10));
    }
  }
  // If none of the above conditions was executed,
  // then the conversion will fail.
  else
    return false;

  // If any of strtod() or strtoll() fails, str will
  // be set to nullptr and this condition will be
  // executed.
  if (str == endptr)
    return false;

  return true;
}

template<typename eT>
bool LoadCSV::LoadNumericCSV(arma::Mat<eT>& x, std::fstream& f)
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
    // Parse the file line by line.
    std::getline(f, lineString);

    if (lineString.size() == 0)
      break;

    lineStream.clear();
    lineStream.str(lineString);

    size_t col = 0;

    while (lineStream.good())
    {
      // Parse each line.
      std::getline(lineStream, token, ',');

      // This will handle loading of both dense and sparse.
      // Initialize tmp_val of type eT with value 0.
      eT tmp_val = eT(0);

      if (ConvertToken<eT>(tmp_val, token))
      {
        x.at(row, col) = tmp_val;
        ++col;
      }
      else
      {
        // Printing failed token and it's location.
        Log::Warn << "Failed to convert token " << token << ", at row " << row << ", column "
                  << col << " of matrix!";

        return false;
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
