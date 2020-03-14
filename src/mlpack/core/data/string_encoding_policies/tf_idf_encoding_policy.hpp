/**
 * @file tf_idf_encoding_policy.hpp
 * @author Jeffin Sam
 *
 * Definition of the TfIdfEncodingPolicy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_ENCODING_POLICIES_TF_IDF_ENCODING_POLICY_HPP
#define MLPACK_CORE_DATA_ENCODING_POLICIES_TF_IDF_ENCODING_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
#include <mlpack/core/data/string_encoding.hpp>

namespace mlpack {
namespace data {

/**
 * Definition of the TfIdfEncodingPolicy class.
 *
 * Tf-idf is a weighting scheme that stands for term-frequency multiplied by
 * inverse document-frequency.
 * The goal of using tf-idf is to scale down the impact of tokens that occur
 * very frequently in a given corpus while using the type of Term-Frequency
 * of a token and that are hence empirically less informative than features
 * that occur in a small fraction of the training corpus.
 * TfIdfEncodingPolicy is used as a helper class for StringEncoding.
 * The encoder assigns a tf-idf number to each unique token and treat 
 * the dataset as categorical. The tokens are labeled in the order of their
 * occurrence in the input dataset.
 */
class TfIdfEncodingPolicy
{
 public:
  /**
   * Enum class used to identify the type of tf encoding.
   *
   * Following are the type definitions:
   * BINARY : binary weighting scheme (0,1)
   * RAW_COUNT : raw count weighting scheme (count of token for every row)
   * TERM_FREQUENCY : term frequency weighting scheme (count / length(row))
   * SUBLINEAR_TF : logarithmic weighting scheme (log(tf) + 1) 
   * 
   */
  enum class TfTypes
  {
    RAW_COUNT,
    BINARY,
    SUBLINEAR_TF,
    TERM_FREQUENCY,
  };

  /**
   * A constructor for the class which is use to set the type of term frequency
   * and also the value for smoothIdf.
   *
   * @param tfType The type of term frequency, The avialbale option are
   *     RAW_COUNT : The count of a specific token
   *     BINARY : 1 if token occurs in document and 0 otherwise;
   *     TERM_FREQUENCY :  Raw_count ÷ (number of words in document)
   *     SUBLINEAR_TF : log(Raw_Count) + 1
   *
   * @param smoothIdf Used to indicate whether to use smooth idf or not.
   */
  TfIdfEncodingPolicy(TfTypes tfType = TfTypes::RAW_COUNT,
                      bool smoothIdf = true) :
                      tfType(tfType),
                      smoothIdf(smoothIdf)
  {
  }

  /**
   * The function initializes the output matrix.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param datasetSize The number of strings in the input dataset.
   * @param maxNumTokens The maximum number of tokens in the strings of the 
                        input dataset.
   * @param dictionarySize The size of the dictionary (not used).
   * @tparam MatType The type of output matrix.
   */
  template<typename MatType>
  static void InitMatrix(MatType& output,
                         size_t datasetSize,
                         size_t /*maxNumTokens*/,
                         size_t dictionarySize)
  {
    output.zeros(datasetSize, dictionarySize);
  }

  /**
   * The function initializes the output matrix.
   * Overloaded function to store result in vector<vector<OutputType>>
   * 
   * @param output Output matrix to store the encoded results.
   * @param datasetSize The number of strings in the input dataset.
   * @param maxNumTokens The maximum number of tokens in the strings of the 
                         input dataset.
   * @param dictionarySize The size of the dictionary (not used).
   * @tparam OutputType The type of output vector.
   */
  template<typename OutputType>
  static void InitMatrix(std::vector<std::vector<OutputType> >& output,
                         size_t datasetSize,
                         size_t /*maxNumTokens*/,
                         size_t dictionarySize)
  {
    output.resize(datasetSize, std::vector<OutputType>(dictionarySize, 0));
  }

  /** 
   * The function performs the TfIdf encoding algorithm i.e. it writes
   * the encoded token to the output.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param value The encoded token.
   * @param row The row number at which the encoding is performed.
   * @param col The row token number at which the encoding is performed.
   * @tparam MatType The type of output matrix.
   */
  template<typename MatType>
  void Encode(MatType& output,
              size_t value,
              size_t row,
              size_t /*col*/)
  {
    double idf, tf;
    if (smoothIdf)
      idf = std::log((output.n_rows + 1) / (1 + idfdict[value - 1])) + 1;
    else
      idf = std::log(output.n_rows / idfdict[value - 1]) + 1;

    if (tfType == TfTypes::TERM_FREQUENCY)
      tf = tokenCount[row][value - 1] / row_size[row];
    else if (tfType == TfTypes::SUBLINEAR_TF)
      tf = std::log(tokenCount[row][value - 1]) + 1;
    else if (tfType == TfTypes::BINARY)
      tf = tokenCount[row][value - 1] > 0 ? 1 : 0;
    else
      tf = tokenCount[row][value - 1];

    output(row, value - 1) =  tf * idf;
  }

  /** 
   * The function performs the TfIdf encoding algorithm i.e. it writes
   * the encoded token to the output.
   * Overload function to accepted vector<vector<OutputType>> as output type.
   *
   * @param output Output matrix to store the encoded results.
   * @param value The encoded token.
   * @param row The row number at which the encoding is performed.
   * @param col The row token number at which the encoding is performed.
   * @tparam OutputType The type of output vector.
   * @tparam OutputType The type of output vector.
   */
  template<typename OutputType>
  void Encode(std::vector<std::vector<OutputType> >& output,
              size_t value,
              size_t row,
              size_t /*col*/)
  {
    double idf, tf;
    if (smoothIdf)
      idf = std::log((output.size() + 1) / (1 + idfdict[value - 1])) + 1;
    else
      idf = std::log(output.size() / idfdict[value - 1]) + 1;

    if (tfType == TfTypes::TERM_FREQUENCY)
      tf = tokenCount[row][value - 1] / row_size[row];
    else if (tfType == TfTypes::SUBLINEAR_TF)
      tf = std::log(tokenCount[row][value - 1]) + 1;
    else if (tfType == TfTypes::BINARY)
      tf = tokenCount[row][value - 1] > 0 ? 1 : 0;
    else
      tf = tokenCount[row][value - 1];

    output[row][value - 1] =  tf * idf;
  }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(tfType);
    ar & BOOST_SERIALIZATION_NVP(smoothIdf);
  }

  /*
   * The function is used to create the datastrcutre will be important to find
   * out idfvalue of words, and then wrtiting the output based on their count.
   *
   * @param row The row number at which the encoding is performed.
   * @param numToken The count of token parsed till now.
   * @param value The encoded token.
   */
  void PreprocessToken(size_t row,
                       size_t /*numTokens*/,
                       size_t value)
  {
    if (row >= tokenCount.size())
    {
      row_size.push_back(0);
      tokenCount.emplace_back();
    }
    tokenCount.back()[value - 1]++;

    if (tokenCount.back()[value - 1] == 1)
      idfdict[value - 1]++;

    row_size.back()++;
  }
 private:
  // Used to store the count of token for each row.
  std::vector<std::unordered_map<size_t, double>> tokenCount;
  // Used to store the idf values.
  std::unordered_map<size_t, double> idfdict;
  // Used to store the number of tokens in each row.
  std::vector<double> row_size;
  // smoothIdf variable to indicate smoothining.
  bool smoothIdf;
  // Type of Term Frequency to use.
  TfTypes tfType;
};

template<typename TokenType>
using TfIdfEncoding = StringEncoding<TfIdfEncodingPolicy,
                          StringEncodingDictionary<TokenType>>;
} // namespace data
} // namespace mlpack

#endif
