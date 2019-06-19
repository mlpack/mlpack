/**
 * @file bow.hpp
 * @author Jeffin Sam
 *
 * Definition of Bag of Words encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_BOW_ENCODING_HPP
#define MLPACK_CORE_DATA_BOW_ENCODING_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <utility>

namespace mlpack {
namespace data {

/**
 * A simple Bow Enocding class
 */
class Bow
{
 public:
  /**
  * A function to create mapping from a given corpus.
  * 
  * @param input Corpus of text to encode.
  * @param tokenizer A function that accepts a boost::string_view as
  *                  an argument and returns a token.
  * This can either be a function pointer or function object or a lamda
  * function. Its return value should be a boost::string_view, a token.
  *
  * Definiation of function should be of type
  * boost::string_view fn()(boost::string_view& str)
  *
  */
  template<typename TokenizerType>
  void CreateMap(const std::string& input, TokenizerType tokenizer);
  /*
  * Default Constructor
  */
  Bow() {}
  /*
  * Copy Constructor.
  */
  Bow(const Bow& oldObject);
  /*
  * Move Constructor.
  */
  Bow(Bow&& oldObject) = default;
  /*
  * Move Assignment Operator.
  */  
  Bow& operator= (Bow&& oldObject) = default;
  /*
  * Assignment Operator.
  */
  Bow& operator= (const Bow& oldObject);
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
  * @param input Vector of strings.
  * @param output Output Matrix to store encoded results (sp_mat or mat).
  * @param tokenizer A function that accepts a boost::string_view as
  *                  an argument and returns a token.
  * This can either be a function pointer or function object or a lamda
  * function. Its return value should be a boost::string_view, a token.
  *
  * Definiation of function should be of type
  * boost::string_view fn()(boost::string_view& str)
  *
  */
  template<typename MatType, typename TokenizerType>
  void Encode(const std::vector<std::string>& input,
              MatType& output, TokenizerType tokenizer);

  //! Modify the originalStrings.
  std::deque<std::string>& OriginalStrings() { return originalStrings; }

  //! Return the originalStrings.
  const std::deque<std::string>& OriginalStrings() const
      { return originalStrings; }

  //! Return the Mappings
  const std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& Mappings() const { return mappings; }

  //! Modify the Mappings.
  std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>>& Mappings() { return mappings; }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);
 private:
  //! A map which stores information about mapping.
  std::unordered_map<boost::string_view, size_t,
      boost::hash<boost::string_view>> mappings;
  //! A deque which holds the original string for map's string_view.
  std::deque<std::string> originalStrings;
}; // class Bow

} // namespace data
} // namespace mlpack

// Include implementation.
#include "bow_encoding_impl.hpp"

#endif
