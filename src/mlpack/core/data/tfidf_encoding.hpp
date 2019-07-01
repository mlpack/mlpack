/**
 * @file tfidf_encoding.hpp
 * @author Jeffin Sam
 *
 * Definition of TfIdf encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_TFIDF_ENCODING_HPP
#define MLPACK_CORE_DATA_TFIDF_ENCODING_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <utility>

using MapType = std::unordered_map<boost::string_view, size_t,
    boost::hash<boost::string_view>>;

namespace mlpack {
namespace data {

/**
 * A simple TfIdf Enocding class.
 *
 * Term frequency–inverse document frequency (Tf-Idf), is a numerical statistic
 * that is intended to reflect how important a word is to a document in a
 * collection or corpus. The encoding here refers to substituting the TFIDF
 * value for the respective word.
 */
class TfIdf
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
  * Definition of function should be of type
  * boost::string_view fn(boost::string_view& str)
  *
  */
  template<typename TokenizerType>
  void CreateMap(std::string& input, TokenizerType tokenizer);
  /**
  * Default Constructor
  */
  TfIdf() {}
  /**
  * Copy Constructor.
  */
  TfIdf(const TfIdf& oldObject);
  /*
  * Move Constructor.
  */
  TfIdf(TfIdf&& oldObject) = default;
  /*
  * Move Assignment Operator.
  */  
  TfIdf& operator= (TfIdf&& oldObject) = default;
  /*
  * Assignment Operator.
  */
  TfIdf& operator= (const TfIdf& oldObject);
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
  * [0 0 0, 0 0 0.1] 
  * User may also provide their custom tokenization rule.
  *
  * @param input Vector of strings.
  * @param output Output Matrix to store encoded results (sp_mat or mat).
  * @param tokenizer A function that accepts a boost::string_view as
  *                  an argument and returns a token.
  * This can either be a function pointer or function object or a lamda
  * function. Its return value should be a boost::string_view, a token.
  *
  * Definition of function should be of type
  * boost::string_view fn(boost::string_view& str)
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

  //! Return the mappings
  const MapType& Mappings() const { return mappings; }

  //! Modify the mapappings.
  MapType& Mappings() { return mappings; }

  //! Return the Idf Values.
  const std::unordered_map<boost::string_view, double,
      boost::hash<boost::string_view>>& IdfValues() const { return idfdict; }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);
 private:
  //! A map which stores information about mapping.
  MapType mappings;
  //! A map which stores informaton about IDF values.
  std::unordered_map<boost::string_view, double,
      boost::hash<boost::string_view>> idfdict;
  //! A deque which holds the original string for map's string_view.
  std::deque<std::string> originalStrings;
}; // class TfIdf

} // namespace data
} // namespace mlpack

// Include implementation.
#include "tfidf_encoding_impl.hpp"

#endif
