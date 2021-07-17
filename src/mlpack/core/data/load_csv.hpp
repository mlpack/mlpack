/**
 * @file core/data/load_csv.hpp
 * @author ThamNgapWei
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
#ifndef MLPACK_CORE_DATA_LOAD_CSV_HPP
#define MLPACK_CORE_DATA_LOAD_CSV_HPP

#include <boost/spirit/include/qi.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <mlpack/core/util/log.hpp>

#include <set>
#include <string>

#include "extension.hpp"
#include "format.hpp"
#include "dataset_mapper.hpp"
#include "types.hpp"

namespace mlpack {
namespace data {

/**
 * Load the csv file.This class use boost::spirit
 * to implement the parser, please refer to following link
 * http://theboostcpplibraries.com/boost.spirit for quick review.
 */
class LoadCSV
{
 public:

  // Do nothing, just a place holder, to be removed later.
  LoadCSV(); 
  /**
   * Construct the LoadCSV object on the given file.  This will construct the
   * rules necessary for loading and attempt to open the file.
   */
  LoadCSV(const std::string& file);

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
  bool LoadCSVFile(MatType& x, std::ifstream& f);

  /**
   * Load the file into the given matrix with the given DatasetMapper object.
   * Throws exceptions on errors.
   *
   * @param inout Matrix to load into.
   * @param infoSet DatasetMapper to use while loading.
   * @param transpose If true, the matrix should be transposed on loading
   *     (default).
   */
  template<typename T, typename PolicyType>
  void Load(arma::Mat<T> &inout,
            DatasetMapper<PolicyType> &infoSet,
            const bool transpose = true)
  {
    CheckOpen();

    if (transpose)
      TransposeParse(inout, infoSet);
    else
      NonTransposeParse(inout, infoSet);
  }

  /**
   * Peek at the file to determine the number of rows and columns in the matrix,
   * assuming a non-transposed matrix.  This will also take a first pass over
   * the data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
   * object will be re-initialized with the correct dimensionality.
   *
   * @param rows Variable to be filled with the number of rows.
   * @param cols Variable to be filled with the number of columns.
   * @param info DatasetMapper object to use for first pass.
   */
  template<typename T, typename MapPolicy>
  void GetMatrixSize(size_t& rows, size_t& cols, DatasetMapper<MapPolicy>& info)
  {
    using namespace boost::spirit;

    // Take a pass through the file.  If the DatasetMapper policy requires it,
    // we will pass everything string through MapString().  This might be useful
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
    {
       ++rows;
    }

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

      if (rows == 1)
      {
        // Extract the number of columns.
	std::pair<int, int> dimen = GetMatSize(inFile);
        cols = dimen.second;
      }

      // I guess this is technically a second pass, but that's ok... still the
      // same idea...
      if (MapPolicy::NeedsFirstPass)
      {
        // In this case we must pass everything we parse to the MapPolicy.
	std::string str(line.begin(), line.end());
        
	for(int i = 0; i < str.size(); i++)
	{
	  // Maybe there is a faster way to parser each element of the string
	  // Also for now it is being considered that delimiter will always
	  // be comma(,)
	  if(str[i] != ',')
	  {
	    std::string cc(1, str[i]);
	    info.template MapFirstPass<T>(std::move(cc), rows - 1);
	  }
	}

      }
    }
  }

  /**
   * Peek at the file to determine the number of rows and columns in the matrix,
   * assuming a transposed matrix.  This will also take a first pass over the
   * data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
   * object will be re-initialized with the correct dimensionality.
   *
   * @param rows Variable to be filled with the number of rows.
   * @param cols Variable to be filled with the number of columns.
   * @param info DatasetMapper object to use for first pass.
   */
  template<typename T, typename MapPolicy>
  void GetTransposeMatrixSize(size_t& rows,
                              size_t& cols,
                              DatasetMapper<MapPolicy>& info)
  {
    using namespace boost::spirit;

    // Take a pass through the file.  If the DatasetMapper policy requires it,
    // we will pass everything string through MapString().  This might be useful
    // if, e.g., the MapPolicy needs to find which dimensions are numeric or
    // categorical.

    // Reset to the start of the file.
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    rows = 0;
    cols = 0;

    std::string line;
    while (std::getline(inFile, line))
    {
      ++cols;

      if (cols == 1)
      {
        // Extract the number of dimensions.
        std::pair<int, int> dimen = GetMatSize(inFile);
        rows = dimen.second;

	// Now that we know the dimensionality, initialize the DatasetMapper.
	info.SetDimensionality(rows);
      }

      // If we need to do a first pas12dds for the DatasetMapper, do it.
      if (MapPolicy::NeedsFirstPass)
      {
        size_t dim = 0;

        // In this case we must pass everything we parse to the MapPolicy.
        std::string str(line.begin(), line.end());

	// Maybe there is a faster way to parser each element of the string
	// Also for now it is being considered that delimiter will always
	// be comma(,)
	for(int i = 0; i < str.size(); i++)
        {
	  if(str[i] != ',')
	  {
	    std::string cc(1, str[i]);
	    info.template MapFirstPass<T>(std::move(cc), dim++);
	  }
        }
      }
    }
  }

 private:
  using iter_type = boost::iterator_range<std::string::iterator>;

  /**
   * Check whether or not the file has successfully opened; throw an exception
   * if not.
   */
  void CheckOpen();

  /**
   * Parse a non-transposed matrix.
   *
   * @param inout Matrix to load into.
   * @param infoSet DatasetMapper object to load with.
   */
  template<typename T, typename PolicyType>
  void NonTransposeParse(arma::Mat<T>& inout,
                         DatasetMapper<PolicyType>& infoSet)
  {
    using namespace boost::spirit;

    // Get the size of the matrix.
    size_t rows, cols;
    GetMatrixSize<T>(rows, cols, infoSet);

    // Set up output matrix.
    inout.set_size(rows, cols);
    size_t row = 0;
    size_t col = 0;

    // Reset file position.
    std::string line;
    inFile.clear();
    inFile.seekg(0, std::ios::beg);

    auto setCharClass = [&](iter_type const &iter)
    {
      std::string str(iter.begin(), iter.end());
      if (str == "\t")
      {
        str.clear();
      }
      boost::trim(str);

      inout(row, col++) = infoSet.template MapString<T>(std::move(str), row);
    };

    while (std::getline(inFile, line))
    {
      // Remove whitespace from either side.
      boost::trim(line);

      // Parse the numbers from a line (ex: 1,2,3,4); if the parser finds a
      // number it will execute the setNum function.
      const bool canParse = qi::parse(line.begin(), line.end(),
          stringRule[setCharClass] % delimiterRule);

      // Make sure we got the right number of rows.
      if (col != cols)
      {
        std::ostringstream oss;
        oss << "LoadCSV::NonTransposeParse(): wrong number of dimensions ("
            << col << ") on line " << row << "; should be " << cols
            << " dimensions.";
        throw std::runtime_error(oss.str());
      }

      if (!canParse)
      {
        std::ostringstream oss;
        oss << "LoadCSV::NonTransposeParse(): parsing error on line " << col
            << "!";
        throw std::runtime_error(oss.str());
      }

      ++row; col = 0;
    }
  }

  /**
   * Parse a transposed matrix.
   *
   * @param inout Matrix to load into.
   * @param infoSet DatasetMapper to load with.
   */
  template<typename T, typename PolicyType>
  void TransposeParse(arma::Mat<T>& inout, DatasetMapper<PolicyType>& infoSet)
  {
    using namespace boost::spirit;

    // Get matrix size.  This also initializes infoSet correctly.
    size_t rows, cols;
    GetTransposeMatrixSize<T>(rows, cols, infoSet);

    // Set the matrix size.
    inout.set_size(rows, cols);

    // Initialize auxiliary variables.
    size_t row = 0;
    size_t col = 0;
    std::string line;
    inFile.clear();
    inFile.seekg(0, std::ios::beg);

    /**
     * This is the parse rule for strings.  When we get a string we have to pass
     * it to the DatasetMapper.
     */
    auto parseString = [&](iter_type const &iter)
    {
      // All parsed values must be mapped.
      std::string str(iter.begin(), iter.end());
      boost::trim(str);

      inout(row, col) = infoSet.template MapString<T>(std::move(str), row);
      ++row;
    };

    while (std::getline(inFile, line))
    {
      // Remove whitespace from either side.
      boost::trim(line);

      // Reset the row we are looking at.  (Remember this is transposed.)
      row = 0;

      // Now use boost::spirit to parse the characters of the line;
      // parseString() will be called when a token is detected.
      const bool canParse = qi::parse(line.begin(), line.end(),
          stringRule[parseString] % delimiterRule);

      // Make sure we got the right number of rows.
      if (row != rows)
      {
        std::ostringstream oss;
        oss << "LoadCSV::TransposeParse(): wrong number of dimensions (" << row
            << ") on line " << col << "; should be " << rows << " dimensions.";
        throw std::runtime_error(oss.str());
      }

      if (!canParse)
      {
        std::ostringstream oss;
        oss << "LoadCSV::TransposeParse(): parsing error on line " << col
            << "!";
        throw std::runtime_error(oss.str());
      }

      // Increment the column index.
      ++col;
    }
  }

  inline std::pair<int, int> GetMatSize(std::ifstream& f);

  //! Spirit rule for parsing.
  boost::spirit::qi::rule<std::string::iterator, iter_type()> stringRule;
  //! Spirit rule for delimiters (i.e. ',' for CSVs).
  boost::spirit::qi::rule<std::string::iterator, iter_type()> delimiterRule;

  //! Extension (type) of file.
  std::string extension;
  //! Name of file.
  std::string filename;
  //! Opened stream for reading.
  std::ifstream inFile;
};

} // namespace data
} // namespace mlpack

#include "load_csv_impl.hpp"

#endif
