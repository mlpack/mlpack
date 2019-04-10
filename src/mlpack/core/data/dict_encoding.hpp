/**
 * @file dict_encoding.hpp
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
  * Default constructor 
  */
  DicitonaryEncoding();

  /**
  * Default Destructor 
  */
  ~DicitonaryEncoding();

  /**
  * A function to create map using a corpus
  * 
  * @param strings Corpus of text to encode
  * @param deliminator Delimiator use to split the corpus
  */
  void CreateMap(std::string strings, char deliminator = ' ');

  /**
  * Overlaoded function to split into characters
  * Since Boost library doesn't have function to split into characters
  *
  * @param strings Corpus of text to encode
  */
  void CreateCharMap(std::string strings);

  /**
  * A function to reset the mapping that is clear all the encodings
  */
  void Reset();

  /**
  * A function to encode strings into characters
  *
  * for example 
  * GATTACA into seven different values
  * The above will be encode to 1233242
  *
  * @param strings Vector of strings
  * @param output Output Matrix to store encoded results
  * @param mapping Mappings use to encode input
  *    
  */
  template<typename eT>
  void CharEncode(const std::vector<std::string>& strings,
              arma::Mat<eT>& output,
              std::unordered_map<std::string, size_t>& mapping);
  /**
  * Overloaded function for a user to avoid passing of mappings
  * Default Data member would be used for mapping
  *
  * @param strings Vector of strings
  * @param output Output Matrix to store encoded results
  *
  */
  template<typename eT>
  void CharEncode(const std::vector<std::string>& strings,
                  arma::Mat<eT>& output);

  /**
  * A fucntion to encode given array of strings using a particular deliminator
  *
  * For example 
  * Vector is :
  * [hello@wow, wow@hello@good] would be encoded using '@' as deliminator as 
  * [1 2 0, 2 1 3] 
  * The function paddes 0 to maintain same sizes across all the rows
  *
  * @param strings Vector of strings
  * @param output Output Matrix to store encoded results
  * @param mapping Mapping use to encode the input
  * @param deliminator Delimnator used to split the strings
  */
  template<typename eT>
  void DictEncode(const std::vector<std::string>& strings,
                  arma::Mat<eT>& output,
                  std::unordered_map<std::string, size_t>& mapping,
                  const char deliminator = ' ');

  /**
  * Overloaded function for the above for user to avoid sending
  * mapping, because it would be handled internally
  *
  * @param strings Vector of strings
  * @param output Output Matrix to store encoded results
  * @param deliminator Delimnator used to split the strings
  */
  template<typename eT>
  void DictEncode(const std::vector<std::string>& strings,
            arma::Mat<eT>& output, const char deliminator = ' ');

  /**
  * A function to encode a single string
  * Can only split using space as deliminator
  *
  * @param string String to encode
  * @param output Output to store encoded values
  * @param dictionary Mapping use to encode values
  */
  template<typename OutputColType = arma::rowvec>
  void DictEncode(const std::string& input, OutputColType&& output,
  std::unordered_map<std::string, size_t>& dictionary);

  /**
  * Overloaded function which provide same functionality as above
  * but allowes a user to not pass mapping, mapping is handled internally
  *
  * @param string String to encode
  * @param output Output to store encoded values
  */
  template<typename OutputColType = arma::rowvec>
  void DictEncode(const std::string& input, OutputColType&& output);

  /**
  * A fucntion to encode given array of strings using a particular deliminator
  *
  * For example 
  * Vector is :
  * [hello@wow, wow@hello@good] would be encoded using '@' as deliminator as 
  * [1 2 , 2 1 3] 
  * The function does not paddes 0 in this case.
  *
  * @param strings Vector of strings
  * @param output Vector of arma::rowvec to store encoded results
  * @param mapping Mapping use to encode the input
  * @param deliminator Delimnator used to split the strings
  */
  template<typename eT>
  void DictEncode(const std::vector<std::string>& strings,
            std::vector<arma::Row<eT>>& output,
            std::unordered_map<std::string, size_t>& mapping,
            const char deliminator = ' ');

  /**
  * Overloaded function for the above for user to avoid sending
  * mapping, because it would be handled internally
  *
  * @param strings Vector of strings
  * @param output Output Matrix to store encoded results
  * @param deliminator Delimnator used to split the strings
  */
  template<typename eT>
  void DictEncode(const std::vector<std::string>& strings,
            std::vector<arma::Row<eT>>& output, const char deliminator = ' ');

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
#include "dict_encoding_impl.hpp"

#endif
