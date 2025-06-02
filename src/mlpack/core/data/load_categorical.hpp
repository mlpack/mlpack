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
 * Armadillo and another for mlpack.
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
#ifndef MLPACK_CORE_DATA_LOAD_CATEGORICAL_HPP
#define MLPACK_CORE_DATA_LOAD_CATEGORICAL_HPP

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
 * Load the csv file. This class contains functions
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
  * rules necessary for loading and will attempt to open the file. This will also
  * initialize the delimiter character for parsing.
  *
  * @param file path of the dataset.
  */
  LoadCSV(const std::string& file, bool fatal) :
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

    CheckOpen(fatal);
  }

  // Functions for Categorical Parse.

  /**
  * Load the file into the given matrix with the given DatasetMapper object.
  *
  * @param inout Matrix to load into.
  * @param infoSet DatasetMapper to use while loading.
  * @param transpose If true, the matrix should be transposed on loading(default).
  * @return false on errors.
  */
  template<typename MatType>
  bool LoadCategoricalCSV(MatType& matrix,
                          TextOptions& opts);

  /**
  * Peek at the file to determine the number of rows and columns in the matrix,
  * assuming a non-transposed matrix.  This will also take a first pass over
  * the data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
  * object will be re-initialized with the correct dimensionality.
  *
  * @param rows Variable to be filled with the number of rows.
  * @param cols Variable to be filled with the number of columns.
  * @param info DatasetMapper object to use for first pass.
  * @return false on errors.
  */
  template<typename T, typename MapPolicy>
  bool InitializeMapper(size_t& rows, size_t& cols,
                        DatasetMapper<MapPolicy>& info,
                        bool fatal);

  /**
  * Peek at the file to determine the number of rows and columns in the matrix,
  * assuming a transposed matrix.  This will also take a first pass over the
  * data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
  * object will be re-initialized with the correct dimensionality.
  *
  * @param rows Variable to be filled with the number of rows.
  * @param cols Variable to be filled with the number of columns.
  * @param info DatasetMapper object to use for first pass.
  * @return false on errors.
  */
  template<typename T, typename MapPolicy>
  bool InitializeTransposeMapper(size_t& rows, size_t& cols,
                                 DatasetMapper<MapPolicy>& info,
                                 bool fatal);

  /**
   * Calculate the number of columns in each row
   * and assign the value to the col. This function
   * will work for categorical data.
   *
   * @param lineStream a single row of data.
   * @param col the number of columns in lineStream.
   * @param delim the delimiter character.
   */
  inline void CategoricalMatColSize(std::stringstream& lineStream, size_t& col,
                                    const char delim);

  // Functions common to both numeric & categorical parser.
  /**
   * Get the size of Categorical matrix.
   *
   * @param f fstream stream to open the data file.
   * @param delim char delimiter charecter.
   */
  inline std::pair<size_t, size_t> CategoricalMatrixSize(std::fstream& f,
      const char delim = ',')
  {
    bool loadOkay = f.good();

    f.clear();
    const std::fstream::pos_type pos1 = f.tellg();

    size_t fnRows = 0;
    size_t fnCols = 0;
    std::string lineString;
    std::stringstream lineStream;
    std::string token;

    while (f.good() && loadOkay)
    {
      // Get a row of data.
      std::getline(f, lineString);
      if (lineString.size() == 0)
        break;

      lineStream.clear();
      lineStream.str(lineString);
      size_t lineNCols = 0;

      // Get number of columns based on the type of data.
      CategoricalMatColSize(lineStream, lineNCols, delim);

      // If there are different number of columns in each
      // row, then the highest number of cols will be
      // considered as the size of the matrix. Missing
      // elements will be filled as 0.
      if (fnCols < lineNCols)
        fnCols = lineNCols;

      ++fnRows;
    }

    f.clear();
    f.seekg(pos1);

    std::pair<size_t, size_t> matSize(fnRows, fnCols);

    return matSize;
  }


 private:
  /**
  * Check whether or not the file has successfully opened; throw an exception
  * if not.
  * @return false on errors.
  */
  inline bool CheckOpen(bool fatal)
  {
    // Check if the file is opening.
    if (!inFile.is_open())
    {
      if (fatal)
        Log::Fatal << "Cannot open file '" << filename
            << "'. File is already open" << std::endl;
      else
        Log::Warn << "Cannot open file '" << filename
            << "'. File is already open" << std::endl;

      return false;
    }

    // Clear format flag.
    inFile.unsetf(std::ios::skipws);
    return true;
  }

  // Functions for Categorical Parse.

  /**
  * Parse a non-transposed matrix.
  *
  * @param input Matrix to load into.
  * @param infoSet DatasetMapper object to load with.
  * @return false on errors.
  */
  template<typename T, typename PolicyType>
  bool NonTransposeParse(arma::Mat<T>& inout,
                         DatasetMapper<PolicyType>& infoSet,
                         bool fatal);

  /**
  * Parse a transposed matrix.
  *
  * @param input Matrix to load into.
  * @param infoSet DatasetMapper to load with.
  * @return false on errors.
  */
  template<typename T, typename PolicyType>
  bool TransposeParse(arma::Mat<T>& inout,
                      DatasetMapper<PolicyType>& infoSet,
                      bool fatal);

  //! Extension (type) of file.
  std::string extension;
  //! Name of file.
  std::string filename;
  //! Opened stream for reading.
  std::fstream inFile;
  //! Delimiter char.
  char delim;
};

} // namespace data
} // namespace mlpack

#include "load_categorical_impl.hpp"

#endif
