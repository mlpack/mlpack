/**
 * @file dictionary_encoding.hpp
 * @author Jeffin Sam
 *
 * Implementation of dictionary encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DICT_ENCODING_HPP
#define MLPACK_CORE_DATA_DICT_ENCODING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
/**
 * A simple Dictionary Enocding class
 */
class DicitonaryEncoding
{
 public:
  /**
  * A function to create map using a corpus.
  * 
  * @param strings Corpus of text to encode.
  * @param delimiter Delimiator use to split the corpus.
  */
  void CreateMap(std::string& strings, std::string delimiter = " ");

  /**
  * Overloaded function to create map using a corpus, which
  * allowes users to provide their own custom tokenization.
  * 
  * @param strings Corpus of text to encode.
  * @param delimiter Delimiator use to split the corpus.
  * @param tokenizer A lamda function providing the rule for tokenization.
  */
  template<typename TokenizerType>
  void CreateMap(std::string& strings, std::string delimiter,
                 TokenizerType tokenizer);

  /**
  * A function to reset the mapping that is clear all the encodings
  */
  void Reset();

  /**
  * A fucntion to encode given array of strings using a particular delimiter,
  * providing custom rule for tokenization.
  *
  * For example 
  * Vector is :
  * [hello@wow, wow@hello@good] would be encoded using '@' as delimiter as 
  * [1 2 0, 2 1 3] 
  * The function paddes 0 to maintain same sizes across all the rows
  * User may also provide their custom tokenization rule.
  *
  * @param strings Vector of strings.
  * @param delimiter Delimiter used to split the strings.
  * @param tokenizer A lamda function providing the rule for tokenization.
  * @param output Output Matrix to store encoded results.
  */
  template<typename MatType, typename TokenizerType>
  void DictEncode(const std::vector<std::string>& strings,
                  const std::string& delimiter,
                  TokenizerType tokenizer,
                  MatType& output);

  /**
  * Overloaded function for the above for user to avoid specifiying
  * tokenization rule and use standard tokenizer class.
  *
  * @param strings Vector of strings.
  * @param output Output Matrix to store encoded results.
  * @param delimiter Delimiter used to split the strings.
  */
  template<typename MatType>
  void DictEncode(const std::vector<std::string>& strings,
                  MatType& output, const std::string& delimiter = " ");

  /**
  * A fucntion to encode given array of strings using a particular delimiter,
  * with custome tokenization.
  *
  * For example 
  * Vector is :
  * [hello@wow, wow@hello@good] would be encoded using '@' as delimiter as 
  * [1 2 , 2 1 3] 
  * The function does not paddes 0 in this case.
  *
  * @param strings Vector of strings.
  * @param delimiter Delimiter used to split the strings.
  * @param tokenizer A lamda function providing the rule for tokenization.
  * @param output Vector of vectors to store encoded results.
  */
  template<typename TokenizerType>
  void DictEncode(const std::vector<std::string>& strings,
            const std::string delimiter,
            TokenizerType tokenizer,
            std::vector<std::vector<int>>& output);

  /**
  * Overloaded function for the above for user to avoid passing custom
  * tokenization rule and use stadandard tokenizer.
  *
  * @param strings Vector of strings.
  * @param delimiter Delimitor used to split the strings.
  * @param output Vector of vectors to store encoded results..
  */
  void DictEncode(const std::vector<std::string>& strings,
                  std::vector<std::vector<int> >& output,
                  const std::string delimiter = " ");

  //! Return the Mappings
  const std::unordered_map<std::string, size_t>& Mappings() const
      { return mappings; }

  //! Modify the Mappings.
  std::unordered_map<std::string, size_t>& Mappings() { return mappings; }

 private:
  //! A map which stores information about mapping.
  std::unordered_map<std::string, size_t>mappings;
}; // class DicitonaryEncoding

} // namespace data
} // namespace mlpack

// Include implementation.
#include "dictionary_encoding_impl.hpp"

#endif
