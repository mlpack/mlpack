/**
 * @file core/data/string_encoding_policies/tf_idf_encoding_policy.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Definition of the TfIdfEncodingPolicy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_TF_IDF_ENCODING_POLICY_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_TF_IDF_ENCODING_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
#include <mlpack/core/data/string_encoding.hpp>

namespace mlpack {
namespace data {

/**
 * Definition of the TfIdfEncodingPolicy class. TfIdfEncodingPolicy is used
 * as a helper class for StringEncoding.
 *
 * Tf-idf is a weighting scheme that takes into account the importance of
 * encoded tokens. The tf-idf statistics is equal to term frequency (tf)
 * multiplied by inverse document frequency (idf).
 * The encoder assigns the corresponding tf-idf value to each token. The order
 * in which the tokens are labeled is defined by the dictionary used by the
 * StringEncoding class. The encoder writes data either in the column-major
 * order or in the row-major order depending on the output data type.
 */
class TfIdfEncodingPolicy
{
 public:
  /**
   * Enum class used to identify the type of the term frequency statistics.
   *
   * The present implementation supports the following types:
   * BINARY           Term frequency equals 1 if the row contains the encoded
   *                  token and 0 otherwise.
   * RAW_COUNT        Term frequency equals the number of times when the encoded
   *                  token occurs in the row.
   * TERM_FREQUENCY   Term frequency equals the number of times when the encoded
   *                  token occurs in the row divided by the total number of
   *                  tokens in the row.
   * SUBLINEAR_TF     Term frequency equals \f$ 1 + log(rawCount), \f$ where
   *                  rawCount is equal to the number of times when the encoded
   *                  token occurs in the row.
   */
  enum class TfTypes
  {
    BINARY,
    RAW_COUNT,
    TERM_FREQUENCY,
    SUBLINEAR_TF,
  };

  /**
   * Construct this using the term frequency type and the inverse document
   * frequency type.
   *
   * @param tfType Type of the term frequency statistics.
   * @param smoothIdf Used to indicate whether to use smooth idf or not.
   *                  If idf is smooth it's calculated by the following formula:
   *                  \f$ idf(T) = \log \frac{1 + N}{1 + df(T)} + 1, \f$ where
   *                  \f$ N \f$ is the total number of strings in the document,
   *                  \f$ T \f$ is the current encoded token, \f$ df(T) \f$
   *                  equals the number of strings which contain the token.
   *                  If idf isn't smooth then the following rule applies:
   *                  \f$ idf(T) = \log \frac{N}{df(T)} + 1. \f$
   */
  TfIdfEncodingPolicy(const TfTypes tfType = TfTypes::RAW_COUNT,
                      const bool smoothIdf = true) :
      tfType(tfType),
      smoothIdf(smoothIdf)
  { }

  /**
   * Clear the necessary internal variables.
   */
  void Reset()
  {
    tokensFrequences.clear();
    numContainingStrings.clear();
    linesSizes.clear();
  }

  /**
   * The function initializes the output matrix. The encoder writes data
   * in the row-major order.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param datasetSize The number of strings in the input dataset.
   * @param * (maxNumTokens) The maximum number of tokens in the strings of the
   *                     input dataset (not used).
   * @param dictionarySize The size of the dictionary.
   */
  template<typename MatType>
  static void InitMatrix(MatType& output,
                         const size_t datasetSize,
                         const size_t /* maxNumTokens */,
                         const size_t dictionarySize)
  {
    output.zeros(dictionarySize, datasetSize);
  }

  /**
   * The function initializes the output matrix. The encoder writes data
   * in the row-major order.
   *
   * Overloaded function to save the result in vector<vector<ElemType>>.
   *
   * @tparam ElemType Type of the output values.
   *
   * @param output Output matrix to store the encoded results.
   * @param datasetSize The number of strings in the input dataset.
   * @param * (maxNumTokens) The maximum number of tokens in the strings of the
   *                     input dataset (not used).
   * @param dictionarySize The size of the dictionary.
   */
  template<typename ElemType>
  static void InitMatrix(std::vector<std::vector<ElemType>>& output,
                         const size_t datasetSize,
                         const size_t /* maxNumTokens */,
                         const size_t dictionarySize)
  {
    output.resize(datasetSize, std::vector<ElemType>(dictionarySize));
  }

  /**
   * The function performs the TfIdf encoding algorithm i.e. it writes
   * the encoded token to the output. The encoder writes data in the
   * column-major order.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param value The encoded token.
   * @param line The line number at which the encoding is performed.
   * @param * (index) The token index in the line.
   */
  template<typename MatType>
  void Encode(MatType& output,
              const size_t value,
              const size_t line,
              const size_t /* index */)
  {
    const typename MatType::elem_type tf =
        TermFrequency<typename MatType::elem_type>(
            tokensFrequences[line][value], linesSizes[line]);

    const typename MatType::elem_type idf =
        InverseDocumentFrequency<typename MatType::elem_type>(
            output.n_cols, numContainingStrings[value]);

    output(value - 1, line) = tf * idf;
  }

  /**
   * The function performs the TfIdf encoding algorithm i.e. it writes
   * the encoded token to the output. The encoder writes data in the
   * row-major order.
   *
   * Overloaded function to accept vector<vector<ElemType>> as the output
   * type.
   *
   * @tparam ElemType Type of the output values.
   *
   * @param output Output matrix to store the encoded results.
   * @param value The encoded token.
   * @param line The line number at which the encoding is performed.
   * @param * (index) The token index in the line.
   */
  template<typename ElemType>
  void Encode(std::vector<std::vector<ElemType>>& output,
              const size_t value,
              const size_t line,
              const size_t /* index */)
  {
    const ElemType tf = TermFrequency<ElemType>(
        tokensFrequences[line][value], linesSizes[line]);

    const ElemType idf = InverseDocumentFrequency<ElemType>(
        output.size(), numContainingStrings[value]);

    output[line][value - 1] = tf * idf;
  }

  /*
   * The function calculates the necessary statistics for the purpose
   * of the tf-idf algorithm during the first pass through the dataset.
   *
   * @param line The line number at which the encoding is performed.
   * @param index The token sequence number in the line.
   * @param value The encoded token.
   */
  void PreprocessToken(const size_t line,
                       const size_t /* index */,
                       const size_t value)
  {
    if (line >= tokensFrequences.size())
    {
      linesSizes.resize(line + 1);
      tokensFrequences.resize(line + 1);
    }

    tokensFrequences[line][value]++;

    if (tokensFrequences[line][value] == 1)
      numContainingStrings[value]++;

    linesSizes[line]++;
  }

  //! Return token frequencies.
  const std::vector<std::unordered_map<size_t, size_t>>&
      TokensFrequences() const { return tokensFrequences; }
  //! Modify token frequencies.
  std::vector<std::unordered_map<size_t, size_t>>& TokensFrequences()
  {
    return tokensFrequences;
  }

  //! Get the number of containing strings depending on the given token.
  const std::unordered_map<size_t, size_t>& NumContainingStrings() const
  {
    return numContainingStrings;
  }

  //! Modify the number of containing strings depending on the given token.
  std::unordered_map<size_t, size_t>& NumContainingStrings()
  {
    return numContainingStrings;
  }

  //! Return the lines sizes.
  const std::vector<size_t>& LinesSizes() const { return linesSizes; }
  //! Modify the lines sizes.
  std::vector<size_t>& LinesSizes() { return linesSizes; }

  //! Return the term frequency type.
  TfTypes TfType() const { return tfType; }
  //! Modify the term frequency type.
  TfTypes& TfType() { return tfType; }

  //! Determine the idf algorithm type (whether it's smooth or not).
  bool SmoothIdf() const { return smoothIdf; }
  //! Modify the idf algorithm type (whether it's smooth or not).
  bool& SmoothIdf() { return smoothIdf; }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(tfType));
    ar(CEREAL_NVP(smoothIdf));
  }

 private:
  /**
   * The function calculates the term frequency statistics.
   *
   * @tparam ValueType Type of the returned value.
   *
   * @param numOccurrences The number of the given token occurrences in
   *                       the line.
   * @param numTokens The total number of tokens in the line.
   */
  template<typename ValueType>
  ValueType TermFrequency(const size_t numOccurrences,
                          const size_t numTokens)
  {
    switch (tfType)
    {
    case TfTypes::BINARY:
      return numOccurrences > 0;
    case TfTypes::RAW_COUNT:
      return numOccurrences;
    case TfTypes::TERM_FREQUENCY:
      return static_cast<ValueType>(numOccurrences) / numTokens;
    case TfTypes::SUBLINEAR_TF:
      return std::log(static_cast<ValueType>(numOccurrences)) + 1;
    default:
      Log::Fatal << "Incorrect term frequency type!";
      return 0;
    }
  }

  /**
   * The function calculates the inverse document frequency statistics.
   *
   * @tparam ValueType Type of the returned value.
   *
   * @param totalNumLines The total number of strings in the input dataset.
   * @param numOccurrences The number of strings in the input dataset
   *                       which contain the current token.
   */
  template<typename ValueType>
  ValueType InverseDocumentFrequency(const size_t totalNumLines,
                                     const size_t numOccurrences)
  {
    if (smoothIdf)
    {
      return std::log(static_cast<ValueType>(totalNumLines + 1) /
          (1 + numOccurrences)) + 1.0;
    }
    else
    {
      return std::log(static_cast<ValueType>(totalNumLines) /
          numOccurrences) + 1.0;
    }
  }

 private:
  //! Used to store the total number of tokens for each line.
  std::vector<std::unordered_map<size_t, size_t>> tokensFrequences;
  /**
   * Used to store the number of strings which contain a token depending
   * on the given token.
   */
  std::unordered_map<size_t, size_t> numContainingStrings;
  //! Used to store the number of tokens in each line.
  std::vector<size_t> linesSizes;
  //! Type of the term frequency scheme.
  TfTypes tfType;
  //! Indicates whether the idf scheme is smooth or not.
  bool smoothIdf;
};

/**
 * A convenient alias for the StringEncoding class with TfIdfEncodingPolicy
 * and the default dictionary for the given token type.
 *
 * @tparam TokenType Type of the tokens.
 */
template<typename TokenType>
using TfIdfEncoding = StringEncoding<TfIdfEncodingPolicy,
                                     StringEncodingDictionary<TokenType>>;
} // namespace data
} // namespace mlpack

#endif
