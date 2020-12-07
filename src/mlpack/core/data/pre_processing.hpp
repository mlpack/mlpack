/**
 * @file core/data/pre_processing_impl.hpp
 * @author Gopi Manohar Tatiraju
 *
 * Generic dataframe-like class for C++ for data pre processing
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATA_PRE_PROCESSING_HPP
#define MLPACK_CORE_DATA_DATA_PRE_PROCESSING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

/**
 * A datafram-like class for pre-processing the dataset 
 * before feeding it to any model. 
 *  
 * @param path Path to the csv file
 * @param headers Headers of the dataset
 * @param numericAttributes Bool array to indicate numeric attributes
 * @param numericData Arma mat of numeric attributes only
 */
class Data {
  private:
    std::string path;
    arma::field<std::string> headers;
    std::vector<bool> numericAttributes;

    bool isDigits(const std::string& str);
    void infoHelper(int lower, int upper);
    void getNumericAttributes();

  public:
    arma::mat data;
    arma::mat numericData;
    /** 
     * Constructor for Data class
     * assign file_path to class variable std::string path
     * load data into arma::mat data
     * extract headers of csv file into std::vector<std::string> headers
     * extract numeric attributes into arma::mat numericData
     */
    Data(std::string file_path);

    /**
     * Get information about the dataset
     * Type of data(current implementation is not final)
     * Index Range
     * List of numeric attributes
     * List of non-numeric attributes
     * List of Data Columns
     */
    void Info();

    /**
     * Describe the dataset in terms of some mathematical functions
     * Count
     * Mean: mean of each numeric attribute
     * Standard Deviation: std of each attribute
     * Minimum
     * Maximum
     * Variance
     */
    void Describe();

    /**
     * Prints top 5 rows of the data 
     */
    void Head();

    /**
     * Prints bottom 5 rows of the data
     */
    void Tail();

    /**
     * For non-numeric attributes, returns the count of 
     * each unique value.
     */
    void valueCounts(int colNumber);

    /**
     * For printing only numeric values 
     */
    void getNumericData();
	};
}
}

#endif