/**
 * @file load_csv.hpp
 * @author ThamNgapWei
 *
 * This is a csv parsers which use to parse the csv file format
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

#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>

#include <set>
#include <string>

#include "extension.hpp"
#include "format.hpp"
#include "dataset_mapper.hpp"

namespace mlpack {
namespace data {

/**
 *Load the csv file.This class use boost::spirit
 *to implement the parser, please refer to following link
 *http://theboostcpplibraries.com/boost.spirit for quick review.
 */
class LoadCSV
{
 public:
  /**
   * Construct the LoadCSV object on the given file.  This will construct the
   * rules necessary for loading and attempt to open the file.
   */
  LoadCSV(const std::string& file);

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
    info = DatasetMapper<MapPolicy>(rows);

    // Now, jump back to the beginning of the file.
    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    rows = 0;

    while (std::getline(inFile, line))
    {
      ++rows;
      // Remove whitespace from either side.
      boost::trim(line);

      if (rows == 1)
      {
        // Extract the number of columns.
        auto findColSize = [&cols](iter_type) { ++cols; };
        qi::parse(line.begin(), line.end(),
            stringRule[findColSize] % delimiterRule);
      }

      // I guess this is technically a second pass, but that's ok... still the
      // same idea...
      if (MapPolicy::NeedsFirstPass)
      {
        // In this case we must pass everything we parse to the MapPolicy.
        auto firstPassMap = [&](const iter_type& iter)
        {
          std::string str(iter.begin(), iter.end());
          boost::trim(str);

          info.template MapFirstPass<T>(std::move(str), rows - 1);
        };

        // Now parse the line.
        qi::parse(line.begin(), line.end(),
            stringRule[firstPassMap] % delimiterRule);
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
      // Remove whitespace from either side.
      boost::trim(line);

      if (cols == 1)
      {
        // Extract the number of dimensions.
        auto findRowSize = [&rows](iter_type) { ++rows; };
        qi::parse(line.begin(), line.end(),
            stringRule[findRowSize] % delimiterRule);

        // Now that we know the dimensionality, initialize the DatasetMapper.
        info.SetDimensionality(rows);
      }

      // If we need to do a first pass for the DatasetMapper, do it.
      if (MapPolicy::NeedsFirstPass)
      {
        size_t dim = 0;

        // In this case we must pass everything we parse to the MapPolicy.
        auto firstPassMap = [&](const iter_type& iter)
        {
          std::string str(iter.begin(), iter.end());
          boost::trim(str);

          info.template MapFirstPass<T>(std::move(str), dim++);
        };

        // Now parse the line.
        qi::parse(line.begin(), line.end(),
            stringRule[firstPassMap] % delimiterRule);
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

#endif
