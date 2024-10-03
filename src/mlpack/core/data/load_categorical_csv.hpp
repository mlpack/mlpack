/**
 * @file core/data/load_categorical_csv.hpp
 * @author Gopi Tatiraju
 *
 * Load a matrix from file. Matrix may contain categorical data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_CATEGORICAL_CSV_HPP
#define MLPACK_CORE_DATA_LOAD_CATEGORICAL_CSV_HPP

#include "load_csv.hpp"

namespace mlpack{
namespace data{

template<typename eT, typename PolicyType>
void LoadCSV::LoadCategoricalCSV(arma::Mat<eT> &inout,
                                 DatasetMapper<PolicyType> &infoSet,
                                 const bool transpose)
{
  CheckOpen();

  if (transpose)
    TransposeParse(inout, infoSet);
  else
    NonTransposeParse(inout, infoSet);
}

inline void LoadCSV::CategoricalMatSize(
    std::stringstream& lineStream, size_t& col, const char delim)
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

template<typename T, typename MapPolicy>
void LoadCSV::InitializeTransposeMapper(size_t& rows, size_t& cols,
                                        DatasetMapper<MapPolicy>& info)
{
  // Take a pass through the file.  If the DatasetMapper policy requires it,
  // we will pass everything as string through MapString().  This might be
  // useful if, e.g., the MapPolicy needs to find which dimensions are numeric
  // or categorical.

  // Reset to the start of the file.
  inFile.clear();
  inFile.seekg(0, std::ios::beg);
  rows = 0;
  cols = 0;

  std::string line;
  while (inFile.good())
  {
    ++cols;

    if (cols == 1)
    {
      // Extract the number of dimensions.
      std::pair<size_t, size_t> dimen = GetMatrixSize<false>(inFile, delim);
      rows = dimen.second;

      if (info.Dimensionality() == 0)
      {
        info.SetDimensionality(rows);
      }
      else if (info.Dimensionality() != rows)
      {
        std::ostringstream oss;
        oss << "data::LoadCSV(): given DatasetInfo has dimensionality "
            << info.Dimensionality() << ", but data has dimensionality "
            << rows;
        throw std::invalid_argument(oss.str());
      }
    }

     std::getline(inFile, line);
     // Remove whitespaces from either side.
     Trim(line);

    // If it's an empty line decrease cols and break.
    if (line.size() == 0)
    {
      --cols;
      continue;
    }

    // If we need to do a first pass for the DatasetMapper, do it.
    if (MapPolicy::NeedsFirstPass)
    {
      // In this case we must pass everything we parse to the MapPolicy.
      size_t dim = 0;
      std::stringstream lineStream;
      std::string token;

      lineStream.clear();
      lineStream.str(line);

      while (lineStream.good())
      {
        std::getline(lineStream, token, delim);
        // Remove whitespace from either side
        Trim(token);

        if (token[0] == '"' && token[token.size() - 1] != '"')
        {
          std::string tok = token;
          while (token[token.size() - 1] != '"')
          {
            tok += delim;
            std::getline(lineStream, token, delim);
            tok += token;
          }
          token = tok;
        }
        info.template MapFirstPass<T>(std::move(token), dim++);
      }
    }
  }
}

template<typename T, typename MapPolicy>
void LoadCSV::InitializeMapper(size_t& rows, size_t& cols,
    DatasetMapper<MapPolicy>& info)
{
  // Take a pass through the file.  If the DatasetMapper policy requires it, we
  // will pass everything as string through MapString().  This might be useful
  // if, e.g., the MapPolicy needs to find which dimensions are numeric or
  // categorical.

  // Reset to the start of the file.
  inFile.clear();
  inFile.seekg(0, std::ios::beg);
  rows = 0;
  cols = 0;

  // First, count the number of rows in the file (this is the dimensionality).
  std::string line;
  while (std::getline(inFile, line))
    ++rows;

  // Reset the DatasetInfo object, if needed.
  if (info.Dimensionality() == 0)
  {
    info.SetDimensionality(rows);
  }
  else if (info.Dimensionality() != rows)
  {
    std::ostringstream oss;
    oss << "data::LoadCSV(): given DatasetInfo has dimensionality "
        << info.Dimensionality() << ", but data has dimensionality "
        << rows;
    throw std::invalid_argument(oss.str());
  }

  // Now, jump back to the beginning of the file.
  inFile.clear();
  inFile.seekg(0, std::ios::beg);
  rows = 0;

  while (std::getline(inFile, line))
  {
    ++rows;
    // Remove whitespaces from either side.
    Trim(line);
    if (rows == 1)
    {
      // Extract the number of columns.
      std::pair<size_t, size_t> dimen = GetMatrixSize<false>(inFile, delim);
      cols = dimen.second;
    }

    // I guess this is technically a second pass, but that's ok... still the
    // same idea...
    if (MapPolicy::NeedsFirstPass)
    {
      std::string str(line.begin(), line.end());
      std::stringstream lineStream;
      std::string token;

      lineStream.clear();
      lineStream.str(line);

      while (lineStream.good())
      {
        std::getline(lineStream, token, delim);
        // Remove whitespace from either side.
        Trim(token);

        if (token[0] == '"' && token[token.size() - 1] != '"')
        {
          std::string tok = token;
          while (token[token.size() - 1] != '"')
          {
            tok += delim;
            std::getline(lineStream, token, delim);
            tok += token;
          }
          token = tok;
        }
        info.template MapFirstPass<T>(std::move(token), rows - 1);
      }
    }
  }
}

template<typename T, typename PolicyType>
void LoadCSV::TransposeParse(arma::Mat<T>& inout,
                             DatasetMapper<PolicyType>& infoSet)
{
  // Get matrix size.  This also initializes infoSet correctly.
  size_t rows, cols;
  InitializeTransposeMapper<T>(rows, cols, infoSet);

  // Set the matrix size.
  inout.set_size(rows, cols);

  // Initialize auxiliary variables.
  size_t row = 0;
  size_t col = 0;
  std::string line;
  inFile.clear();
  inFile.seekg(0, std::ios::beg);

  while (std::getline(inFile, line))
  {
    // Remove whitespaces from either side.
    Trim(line);
    // Reset the row we are looking at.  (Remember this is transposed.)
    row = 0;
    std::stringstream lineStream;
    std::string token;
    lineStream.clear();
    lineStream.str(line);

    while (lineStream.good())
    {
      std::getline(lineStream, token, delim);
      // Remove whitespaces from either side.
      Trim(token);

      if (token[0] == '"' && token[token.size() - 1] != '"')
      {
        // First part of the string.
        std::string tok = token;
        while (token[token.size() - 1] != '"')
        {
          tok += delim;
          std::getline(lineStream, token, delim);
          tok += token;
        }
        token = tok;
      }
      inout(row, col) = infoSet.template MapString<T>(std::move(token), row);
      row++;
    }

    // Make sure we got the right number of rows.
    if (row != rows)
    {
      std::ostringstream oss;
      oss << "LoadCSV::TransposeParse(): wrong number of dimensions (" << row
          << ") on line " << col << "; should be " << rows << " dimensions.";
      throw std::runtime_error(oss.str());
    }
    // Increment the column index.
    ++col;
  }
}

template<typename T, typename PolicyType>
void LoadCSV::NonTransposeParse(arma::Mat<T>& inout,
                                DatasetMapper<PolicyType>& infoSet)
{
  // Get the size of the matrix.
  size_t rows, cols;
  InitializeMapper<T>(rows, cols, infoSet);

  // Set up output matrix.
  inout.set_size(rows, cols);
  size_t row = 0;
  size_t col = 0;

  // Reset file position.
  std::string line;
  inFile.clear();
  inFile.seekg(0, std::ios::beg);

  while (std::getline(inFile, line))
  {
    // Remove whitespaces from either side.
    Trim(line);

    std::stringstream lineStream;
    std::string token;

    lineStream.clear();
    lineStream.str(line);

    while (lineStream.good())
    {
      if (token == "\t")
        token.clear();

      std::getline(lineStream, token, delim);
      // Remove whitespace from either side.
      Trim(token);

      if (token[0] == '"' && token[token.size() - 1] != '"')
      {
        std::string tok = token;
        while (token[token.size() - 1] != '"')
        {
          tok += delim;
          std::getline(lineStream, token, delim);
          tok += token;
        }
        token = tok;
      }
      inout(row, col++) = infoSet.template MapString<T>(std::move(token), row);
    }

    // Make sure we got the right number of rows.
    if (col != cols)
    {
      std::ostringstream oss;
      oss << "LoadCSV::NonTransposeParse(): wrong number of dimensions ("
          << col << ") on line " << row << "; should be " << cols
          << " dimensions.";
      throw std::runtime_error(oss.str());
    }
    ++row; col = 0;
  }
}

} //namespace data
} //namespace mlpack

#endif
