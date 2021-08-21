/**
 * @file core/data/load_csv.hpp
 * @author ThamNgapWei
 * @author Conrad Sanderson
 * @author Gopi M. Tatiraju
 *
 * This csv parser is designed by taking reference from
 * armadillo's csv parser. In this mlpack's version, all
 * the arma dependencies were removed or replaced
 * accordingly, making the parser totally independent of
 * armadillo.
 *
 * As the implementation is inspired from Armadillo it
 * is necessary to add two different licenses. One for
 * Armadillo and other for mlpack.
 *
 * https://gitlab.com/conradsnicta/armadillo-code/-/blob/10.5.x/include/armadillo_bits/diskio_meat.hpp
 *
 * The original Armadillo parser is licensed under the
 * BSD-compatible Apache license, shown below:
 *
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

#include <mlpack/core/util/log.hpp>
#include <set>
#include <string>

#include "string_algorithms.hpp"
#include "extension.hpp"
#include "format.hpp"
#include "dataset_mapper.hpp"
#include "types.hpp"

namespace mlpack {
namespace data {

/**
 * Load the csv file.This class contains fucntions
 * to load numeric and categorical data.
 */
class LoadCSV
{
 public:

  LoadCSV()
  {
    // Nothing to do here.
    // To initialize the class object.
  }

  /**
  * Construct the LoadCSV object on the given file.  This will construct the
  * rules necessary for loading and attempt to open the file. This will also
  * initialize the delimiter character for parsing.
  *
  * @param file path of the dataset
  */
  LoadCSV(const std::string& file) :
    extension(Extension(file)),
    filename(file),
    inFile(file)
  {
    if (extension == "csv")
    {
      delim = ',';
    }
    else if (extension == "tsv")
    {
      delim = '\t';
    }
    else if (extension == "txt")
    {
      delim = ' ';
    }

    CheckOpen();
  }

  // Fucntions for Numeric Parser

  /**
  * Returns a bool value showing whether data was loaded successfully or not.
  *
  * Parses a csv file and loads the data into a given matrix. In the first pass,
  * the function will determine the number of cols and rows in the given file.
  * Once the rows and cols are fixed we initialize the matrix with zeros. In 
  * the second pass, the function converts each value to required datatype
  * and sets it equal to val. 
  *
  * @param x Matrix in which data will be loaded
  * @param f File stream to access the data file
  */
  template<typename eT>
  bool LoadNumericCSV(arma::Mat<eT>& x, std::fstream& f);

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
  template<typename eT>
  bool ConvertToken(eT& val, const std::string& token);

  /**
   * Caluculate number of columns in each row
   * and assign the value to the col. This fucntion
   * will work only for numeric data.
   *
   * @param lineStream a single row of data
   * @param col number of columns in lineStream
   * @param delim delimiter character
   */
  inline void NumericMatSize(std::stringstream& lineStream, size_t& col,
                             const char delim); 

  // Functions for Categorical Parse

  /**
  * Load the file into the given matrix with the given DatasetMapper object.
  * Throws exceptions on errors.
  *
  * @param inout Matrix to load into.
  * @param infoSet DatasetMapper to use while loading.
  * @param transpose If true, the matrix should be transposed on loading
  *     (default).
  */
  template<typename eT, typename PolicyType>
  void LoadCategoricalCSV(arma::Mat<eT> &inout,
                          DatasetMapper<PolicyType> &infoSet,
                          const bool transpose = true);

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
  void InitializeMapper(size_t& rows, size_t& cols,
                        DatasetMapper<MapPolicy>& info);

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
  void InitializeTransposeMapper(size_t& rows, size_t& cols,
                              DatasetMapper<MapPolicy>& info);

  /**
   * Caluculate number of columns in each row
   * and assign the value to the col. This fucntion
   * will work for categorical data.
   *
   * @param lineStream a single row of data
   * @param col number of columns in lineStream
   * @param delim delimiter character
   */
  inline void CategoricalMatSize(std::stringstream& lineStream, size_t& col,
                                 const char delim); 

  // Functions common to both numeric & categorical parser

  /**
   * Get the size of the matrix. Based on isNumeric the fucntion can be used
   * for both numeric_parse and categorical_parse.
   *
   * @param f fstream stream to open the data file
   * @param delim char delimiter charecter
   */
  template<bool isNumeric>
  inline std::pair<size_t, size_t> GetMatrixSize(std::fstream& f,
                                                 const char delim = ',')
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
      // Get a row of data
      std::getline(f, lineString);
      if (lineString.size() == 0)
        break;

      lineStream.clear();
      lineStream.str(lineString);

      size_t line_n_cols = 0;

      // Get number of columns based on the type of data
      if (isNumeric)
        NumericMatSize(lineStream, line_n_cols, delim);
      else
        CategoricalMatSize(lineStream, line_n_cols, delim);

      // If there are different number of columns in each
      // row, then the highest number of cols will be
      // considered as the size of the matrix. Missing
      // elements will be filled as 0
      if (f_n_cols < line_n_cols)
        f_n_cols = line_n_cols;

      ++f_n_rows;
    }

    f.clear();
    f.seekg(pos1);

    std::pair<size_t, size_t> mat_size(f_n_rows, f_n_cols);

    return mat_size;
  }


 private:

  /**
  * Check whether or not the file has successfully opened; throw an exception
  * if not.
  */
  inline void CheckOpen()
  {
    // check if file is opening
    if (!inFile.is_open())
    {
      std::ostringstream oss;
      oss << "Cannot open file '" << filename << "'. " << std::endl;
      // throw an exception if file is not opening
      throw std::runtime_error(oss.str());
    }

    // clear format flag 
    inFile.unsetf(std::ios::skipws);
  }

  // Fucntions for Categorical Parse

  /**
  * Parse a non-transposed matrix.
  *
  * @param inout Matrix to load into.
  * @param infoSet DatasetMapper object to load with.
  */
  template<typename T, typename PolicyType>
  void NonTransposeParse(arma::Mat<T>& inout,
                         DatasetMapper<PolicyType>& infoSet);

  /**
  * Parse a transposed matrix.
  *
  * @param inout Matrix to load into.
  * @param infoSet DatasetMapper to load with.
  */
  template<typename T, typename PolicyType>
  void TransposeParse(arma::Mat<T>& inout, DatasetMapper<PolicyType>& infoSet);

  //! Extension (type) of file.
  std::string extension;
  //! Name of file.
  std::string filename;
  //! Opened stream for reading.
  std::fstream inFile;
  //! Delimiter char
  char delim;
};

} // namespace data
} // namespace mlpack

#include "load_numeric_csv.hpp"
#include "load_categorical_csv.hpp"

#endif
